import imageio.v3 as iio
import cv2

stream_url = "udp://@:8554"  # Replace with your VLC stream URL

# Open the stream with imageio
try:
    for frame in iio.imiter(stream_url, plugin="ffmpeg"):
        # Convert the frame to OpenCV format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow("Video Stream", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")

cv2.destroyAllWindows()
