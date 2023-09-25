import torch
import pytorchvideo

from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from pytorchvideo.data.utils import MultiProcessSampler

def get_labeled_video_paths(file_path):
    # Convert csv file of format "video_path1, video_path2, label" to list of tuples
    # of format (video_path1, video_path2, label)
    labeled_video_paths = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labeled_video_paths.append(tuple(line.split(',')))
    return labeled_video_paths

class SyncDataset(LabeledVideoDataset):
    """
    Extend the normal LabeledVideoDataset to allow two videos as input and stack them along channels.
    """

    def __init__(self, new_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labeled_video_paths = get_labeled_video_paths(new_path)

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <stacked_video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video1, video2, label, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    video_path1, video_path2, label = self._labeled_videos[video_index]
                    video1 = self.video_path_handler.video_from_path(
                        video_path1,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    video2 = self.video_path_handler.video_from_path(
                        video_path2,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video1, video2, label, video_index)
                except Exception as e:
                    print(
                        "Failed to load videos {} and {} with error: {}; trial {}".format(
                            video_path1,
                            video_path2,
                            e,
                            i_try,
                        )
                    )
                    continue
            # Set video1 to be the short of the two videos, flip sign of label accordingly
            if video1.duration > video2.duration:
                video1, video2 = video2, video1
                label = -label

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(
                self._next_clip_start_time, video1.duration
            )

            # Only load the clip once and reuse previously stored clip if there are multiple
            # views for augmentations to perform on the same clip.
            if aug_index == 0:
                self._loaded_clip1 = video1.get_clip(clip_start, clip_end)
                self._loaded_clip2 = video2.get_clip(clip_start, clip_end)

            self._next_clip_start_time = clip_end

            video_is_null = (
                self._loaded_clip1 is None or self._loaded_clip1["video"] is None or
                self._loaded_clip2 is None or self._loaded_clip2["video"] is None
            )
            if is_last_clip or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._next_clip_start_time = 0.0

                if video_is_null:
                    print(
                        "Failed to load clip {} and {}; trial {}".format(video1.name, video2.name, i_try)
                    )
                    continue

            frames1 = self._loaded_clip1["video"]
            frames2 = self._loaded_clip2["video"]
            frames = torch.cat((frames1, frames2), dim=0)
            print("Converting shapes {} and {} into shape {}.".format(frames1.shape, frames2.shape, frames.shape))
            sample_dict = {
                "video": frames,
                "video1_name": video1.name,
                "video2_name": video2.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

                # User can force dataset to continue by returning None in transform.
                if sample_dict is None:
                    continue

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

def get_sync_dataset(*args, **kwargs):
    return SyncDataset(args[0], *args, **kwargs)
        