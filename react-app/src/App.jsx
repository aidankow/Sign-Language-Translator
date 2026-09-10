import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import CustomWebcam from "./CustomWebcam";
import "./App.css";

import * as modelModule from "./model"; 

const labels = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
  'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
  'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X',
  'Y', 'Z'
];

function extractNormLocZoom(landmarks, handedness) {
  // Norm_Loc
  const wrist = landmarks[0];

  const xMultiplier = (handedness === "Left") ? 1 : -1;

  const centered = landmarks.map(lm => ({
    x: xMultiplier * (lm.x - wrist.x),
    y: lm.y - wrist.y,
    z: lm.z - wrist.z
  }));

  // Norm_Zoom
  let maxVal = 0;
  centered.forEach(pt => {
    maxVal = Math.max(maxVal, Math.abs(pt.x), Math.abs(pt.y));
  });
  maxVal = Math.max(maxVal, 1e-6); // Prevent division by zero

  const features = [];
  centered.forEach(pt => {
    features.push(pt.x / maxVal, pt.y / maxVal, pt.z / maxVal);
  });

  return features;
}

function App() {
  const webcamRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const [liveText, setLiveText] = useState("...");
  const [translation, setTranslation] = useState("");
  const [errorLog, setErrorLog] = useState(null);

  const lastSignRef = useRef("");
  const consecutiveFramesRef = useRef(0);
  const hasAddedRef = useRef(false);
  const STABILITY_THRESHOLD = 15;

  const predictFunc = modelModule.score || modelModule.default;

  const handleKeyDown = (e) => {
    if (e.key !== "Backspace" && e.key !== "Delete" && e.key) {
      e.preventDefault();
    }
  };

  useEffect(() => {
    async function runMediaPipe() {
      try {
        if (!predictFunc) {
          setErrorLog("Model function score not found in model.js");
          return;
        }

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "CPU",
          },
          runningMode: "IMAGE",
          numHands: 1,
        });

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
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const loop = async () => {
      try {
        if (
          webcamRef.current &&
          webcamRef.current.video &&
          handLandmarkerRef.current &&
          predictFunc
        ) {
          const video = webcamRef.current.video;
          
          if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
            lastVideoTime = video.currentTime;

            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Use .detect() with a canvas element instead of detectForVideo
            const results = handLandmarkerRef.current.detect(canvas);

            if (results.landmarks && results.landmarks.length > 0) {
              const handLandmarks = results.landmarks[0];
              const handedness = results.handedness[0][0].categoryName;
              
              const features = extractNormLocZoom(handLandmarks, handedness);
              const prediction = predictFunc(features);

              const maxScore = Math.max(...prediction);
              const maxIndex = prediction.indexOf(maxScore);

              const detectedLabel = maxScore > 0.55 ? labels[maxIndex] : "Uncertain";

              setLiveText(`${detectedLabel} (${maxScore.toFixed(2)})`);

              if (detectedLabel !== "Uncertain") {
                if (detectedLabel === lastSignRef.current) {
                  if (!hasAddedRef.current) {
                    consecutiveFramesRef.current += 1;
                    if (consecutiveFramesRef.current >= STABILITY_THRESHOLD) {
                      if (detectedLabel === 'Space') {
                        setTranslation((prev) => prev + ' ');
                      } else {
                        setTranslation((prev) => prev + detectedLabel);
                      }
                      hasAddedRef.current = true;
                    }
                  }
                } else {
                  lastSignRef.current = detectedLabel;
                  consecutiveFramesRef.current = 1;
                  hasAddedRef.current = false;
                }
              } else {
                lastSignRef.current = "";
                consecutiveFramesRef.current = 0;
                hasAddedRef.current = false;
              }
            } else {
              setLiveText("No hand detected");
              lastSignRef.current = "";
              consecutiveFramesRef.current = 0;
              hasAddedRef.current = false;
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
          <label id="live-translation">Detected: {liveText}</label>
          <textarea
            id="translation"
            value={translation}
            onChange={(e) => setTranslation(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Your translated sign language will appear here..."
          />
        </div>
      </div>
    </div>
  );
}

export default App;