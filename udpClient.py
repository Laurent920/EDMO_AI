import socket
import av
import cv2
import io

def udp_mpeg_ts_client(port=8554, buffer_size=65536):
    """
    A UDP client that receives MPEG TS data, decodes it, and displays the video frames.
    """
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)  # Increase buffer size
    udp_socket.bind(('0.0.0.0', port))
    print(f"Listening for MPEG TS data on UDP port {port}...")

    buffer = io.BytesIO()  # In-memory buffer to hold MPEG TS data

    try:
        while True:
            # Receive UDP packet
            packet, _ = udp_socket.recvfrom(buffer_size)

            # Write the packet to the in-memory buffer
            buffer.write(packet)

            # Seek to the beginning for reading
            buffer.seek(0)

            try:
                # Open the MPEG TS container with enhanced probing options
                container = av.open(
                    buffer,
                    format='mpegts',
                    options={
                        "probesize": str(5000000),  # Increase probe size to analyze more data
                        "analyzeduration": str(5000000),  # Increase analysis duration
                    }
                )

                for packet in container.demux():
                    if packet.stream.type != 'video':
                        continue

                    for frame in packet.decode():
                        # Convert the AV frame to OpenCV format
                        img = frame.to_ndarray(format='bgr24')

                        # Display the frame
                        cv2.imshow('MPEG TS Stream', img)

                        # Exit on 'q' key press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Exiting...")
                            return

            except av.AVError as e:
                print(f"Decoding error: {e}")
            
            # Clear the buffer for the next packet
            buffer.seek(0)
            buffer.truncate(0)
    except KeyboardInterrupt:
        print("\nUDP client stopped.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        udp_socket.close()
        cv2.destroyAllWindows()

# Run the UDP client
udp_mpeg_ts_client(port=8554)