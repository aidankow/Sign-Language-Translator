import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import CustomWebcam from "./CustomWebcam";
import "./App.css";

import * as modelModule from "./model"; 

const labels = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
  'K','L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
  'T', 'U', 'V', 'W', 'X', 'Y', 'J', 'Z'
];

function App() {
  const webcamRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const [translation, setTranslation] = useState("Initializing...");
  const [handSize, setHandSize] = useState(0);
  const [errorLog, setErrorLog] = useState(null);

  const predictFunc = modelModule.score || modelModule.default;

  useEffect(() => {
    async function runMediaPipe() {
      try {
        if (!predictFunc) {
          setErrorLog("Model function score not found in model.js");
          return;
        }

        setTranslation("Loading MediaPipe WASM...");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        setTranslation("Creating Hand Landmarker...");
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });

        setTranslation("Ready. Show your sign!");
        predictionLoop();
      } catch (err) {
        console.error(err);
        setErrorLog(err.message);
        setTranslation("Error during initialization.");
      }
    }

    runMediaPipe();
  }, [predictFunc]);

  const predictionLoop = () => {
    let lastVideoTime = -1;

    const loop = async () => {
      try {
        if (
          webcamRef.current &&
          webcamRef.current.video &&
          handLandmarkerRef.current &&
          predictFunc
        ) {
          const video = webcamRef.current.video;
          
          if (video.currentTime !== lastVideoTime && video.readyState === 4) {
            lastVideoTime = video.currentTime;
            const results = handLandmarkerRef.current.detectForVideo(video, performance.now());

            if (results.landmarks && results.landmarks.length > 0) {
              const landmarks = results.landmarks[0];
              const wrist = landmarks[0];
              const middleMcp = landmarks[9];
              
              const currentSize = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y);
              setHandSize(currentSize);

              // const MAX_SIZE = 0.3;
              // const MIN_SIZE = 0.16;

              // if (currentSize > MAX_SIZE) {
              //   setTranslation("⚠️ Move further away from the camera");
              //   requestAnimationFrame(loop);
              //   return;
              // } else if (currentSize < MIN_SIZE) {
              //   setTranslation("⚠️ Move closer to the camera");
              //   requestAnimationFrame(loop);
              //   return;
              // }

              const features = landmarks.flatMap((pt) => [
                pt.x - wrist.x, 
                pt.y - wrist.y, 
                pt.z - wrist.z
              ]);

              const predictionArray = predictFunc(features);

              const maxScore = Math.max(...predictionArray);
              const maxIndex = predictionArray.indexOf(maxScore);

              const detectedLabel = maxScore > 0.3 ? labels[maxIndex] : "Uncertain";

              setTranslation(`Detected: ${detectedLabel} (Score: ${maxScore.toFixed(2)})`);
            }
          }
        }
      } catch (loopErr) {
        console.error("Prediction loop error:", loopErr);
      }
      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  };

  if (errorLog) {
    return (
      <div style={{ padding: "40px", color: "red", textAlign: "center" }}>
        <h2>App Crashed:</h2>
        <p>{errorLog}</p>
        <p>Check your browser console (F12) for details.</p>
      </div>
    );
  }

  return (
    <div className="v-box">
      <h2>Sign Language Translator</h2>

      <div className="h-box">
        <CustomWebcam ref={webcamRef} />
        
        <div className="text-panel">
          <label htmlFor="translation">Translated Text:</label>
          <textarea
            id="translation"
            value={translation}
            placeholder="Your translated sign language will appear here..."
            readOnly
          />
        </div>
      </div>

      <div style={{ marginTop: "10px", fontSize: "14px", color: "#666" }}>
        Live Hand Size Value: <strong>{handSize.toFixed(3)}</strong>
      </div>
    </div>
  );
}

export default App;