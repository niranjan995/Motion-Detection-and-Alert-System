from twilio.rest import Client

def send_sms():
    # Load environment variables from a .env file
    
    account_sid='AC97f36b1256ee153191d0c4af81f737b9'
    auth_token='bfbdc16e67668904c718e3b638c3c687'
    from_phone='+12769001969'
    to_phone='+919164350588'
    #Add twilio account info

    client = Client(account_sid, auth_token)
    message = client.messages.create(
            body="unauthorized person is detected",
            from_=from_phone,
            to=to_phone
        )
    print(f"Message sent successfully: {message.sid}")
    
import cv2
import numpy as np
import time

def main():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

    # Initialize variables
    motion_detected = False
    video_writer = None
    last_motion_time = time.time()

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Get frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Parameters for motion detection
    min_contour_area = 5000  # Adjust according to your needs

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize static background
        if 'static_back' not in locals():
            static_back = gray
            continue

        # Compute absolute difference between static background and current frame
        frame_delta = cv2.absdiff(static_back, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if motion_detected:
            send_sms()
            if video_writer is None:
                # Start recording
                video_filename = f"motion_{int(time.time())}.avi"
                video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))
            last_motion_time = time.time()
        else:
            if video_writer is not None and time.time() - last_motion_time >= 3:
                # Stop recording
                video_writer.release()
                video_writer = None

        if video_writer is not None:
            video_writer.write(frame)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
