import subprocess
from fractions import Fraction
from typing import Optional

import av
import numpy as np


def read_video(input_fn: str, frame_format: Optional[str] = "rgb24") -> np.array:
    """
    read video to 4D numpy array with pyav

    input_fn: The input file name, e.g. 'input.mp4'
    frame_format: The format of the input frames, default to 'rgb24', use `ffmpeg -pix_fmts` to list all available formats
    """
    # Open the video file
    container = av.open(input_fn)

    # Get the video stream
    stream = container.streams.video[0]

    # Initialize a list to store the frames
    frames = []

    # Iterate through the packets in the video stream
    for packet in container.demux(stream):
        # Decode the packet
        for frame in packet.decode():
            # Convert the frame to a NumPy array
            frame_data = frame.to_ndarray(format=frame_format)
            # Add the frame to the list
            frames.append(frame_data)

    # Stack the frames along the first axis to create a 4D array
    video = np.stack(frames, axis=0)

    return video  # (num_frames, height, width, num_channels)


def write_video(
    output_fn: str,
    frames: np.array,
    sample_rate: Optional[int] = 30,
    output_codec: Optional[str] = "h264",
    output_options: Optional[dict] = {},
    pix_fmt: Optional[str] = "yuv420p",
    frame_format: Optional[str] = "rgb24",
) -> None:
    """
    write video with pyav

    output_fn: The output file name, e.g. 'output.mp4'
    frames: The frames to write to the output file, expected to be 4D numpy array in the format (t, h, w, c)
    sample_rate: The sample rate of the output video, default to 30
    output_codec: The output codec, default to 'h264', use `ffmpeg -codecs` to list all available codecs
    output_options: The output options, default to {}
    pix_fmt: The pixel format of the output video, default to 'yuv420p', use `ffmpeg -pix_fmts` to list all available formats
    frame_format: The format of the input frames, default to 'rgb24', use `ffmpeg -pix_fmts` to list all available formats
    """
    # Open the output file
    container = av.open(output_fn, "w")

    # Set the output format to H.264
    video_stream = container.add_stream(output_codec, options=output_options)
    video_stream.pix_fmt = pix_fmt

    # Set the frame rate and frame size
    video_stream.rate = sample_rate
    video_stream.width = frames.shape[2]
    video_stream.height = frames.shape[1]
    video_stream.time_base = Fraction(1, sample_rate)

    # Write the frames to the output file
    for frame in frames:
        # Encode and write the video frame
        video_frame = av.VideoFrame.from_ndarray(frame, format=frame_format)
        for packet in video_stream.encode(video_frame):
            container.mux(packet)

    # Flush the encoders
    for packet in video_stream.encode():
        container.mux(packet)

    # Close the output file
    container.close()
