import Webcam from "react-webcam";

function CustomWebcam(props, ref) {
    const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  return (
    <div className="webcam-container">
      <Webcam
        ref={ref}
        audio={false}
        mirrored={true}
        autoPlay={true}
        playsInline={true}
        muted={true}
        videoConstraints={videoConstraints}
        style={{ pointerEvents: "none" }}
      />
    </div>
  );
}

export default forwardRef(CustomWebcam)