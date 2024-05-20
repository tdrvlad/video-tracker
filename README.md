# Video-Tracker 🎬 (with Multiple streams)
Video tracker demo that can handle multiple different streams at once.

![](data/result.gif)
## Multiple Streams 🔀
In a practical application, a video tracking solution might have to handle multiple different stream at once.
For example, the Tracker might be called with short 5-second videos, in order to minimize the latency of processing long videos.

In this case, multiple different streams might be processed at the same time.
The Tracker requires persistent information across one stream in order to correctly maintain the identifications.

One solution would be to instantiate multiple Trackers in parallel threads.
However, this means that multiple instances of the YOLO model are created which is inefficient.

In this approach, we only keep a cache of the tracking output, thus using only a single YOLO model instance.
