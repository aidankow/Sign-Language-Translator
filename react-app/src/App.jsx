import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import CustomWebcam from "./CustomWebcam";
import "./App.css";

import * as modelModule from "./model"; 

const labels = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
  'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
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
            delegate: "GPU",
          },
          runningMode: "VIDEO",
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
              const handLandmarks = results.landmarks[0];

              const handedness = results.handedness[0][0].categoryName;
              
              const features = extractNormLocZoom(handLandmarks, handedness);
              const prediction = predictFunc(features);

              const maxScore = Math.max(...prediction);
              const maxIndex = prediction.indexOf(maxScore);

              const detectedLabel = maxScore > 0.55 ? labels[maxIndex] : "Uncertain";

              setLiveText(`${detectedLabel} (${maxScore.toFixed(2)})`);

              // --- SYMBOL COLLECTION LOGIC ---
              if (detectedLabel !== "Uncertain") {
                if (detectedLabel === lastSignRef.current) {
                  if (!hasAddedRef.current) {
                    consecutiveFramesRef.current += 1;
                    if (consecutiveFramesRef.current >= STABILITY_THRESHOLD) {
                      setTranslation((prev) => prev + detectedLabel);
                      hasAddedRef.current = true; // Lock so it only adds once per hold
                    }
                  }
                } else {
                  // User switched to a new sign
                  lastSignRef.current = detectedLabel;
                  consecutiveFramesRef.current = 1;
                  hasAddedRef.current = false;
                }
              } else {
                // Hand is uncertain / out of frame
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