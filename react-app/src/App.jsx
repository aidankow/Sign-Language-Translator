import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import CustomWebcam from "./CustomWebcam";
import "./App.css";

import * as modelModule from "./model"; 
// Import your local asset here (adjust relative path if App.jsx is nested differently)
import fingerspellingImg from "./assets/fingerspelling.png";

const labels = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
  'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
  'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X',
  'Y', 'Z'
];

function extractNormLocZoom(landmarks, handedness) {
  const wrist = landmarks[0];
  const xMultiplier = (handedness === "Left") ? 1 : -1;

  const centered = landmarks.map(lm => ({
    x: xMultiplier * (lm.x - wrist.x),
    y: lm.y - wrist.y,
    z: lm.z - wrist.z
  }));

  let maxVal = 0;
  centered.forEach(pt => {
    maxVal = Math.max(maxVal, Math.abs(pt.x), Math.abs(pt.y));
  });
  maxVal = Math.max(maxVal, 1e-6);

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
  const [showChart, setShowChart] = useState(false);
  
  const [isHwAccelerated, setIsHwAccelerated] = useState(() => {
    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
      return !!gl;
    } catch (e) {
      return false;
    }
  });

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
    if (!isHwAccelerated) return;

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
        if (err.message && (err.message.includes("kGpuService") || err.message.includes("activeTexture"))) {
          setIsHwAccelerated(false);
        } else {
          setErrorLog(err.message);
        }
      }
    }

    runMediaPipe();
  }, [isHwAccelerated, predictFunc]);

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

  if (!isHwAccelerated) {
    return (
      <div style={{ padding: "40px", textAlign: "center", maxWidth: "600px", margin: "auto", fontFamily: "sans-serif" }}>
        <div style={{ background: "#fff3cd", border: "1px solid #ffeeba", color: "#856404", padding: "20px", borderRadius: "8px" }}>
          <h3 style={{ margin: "0 0 10px 0" }}>⚠️ Graphics Acceleration Required</h3>
          <p style={{ lineHeight: "1.5" }}>
            This application requires <b>Graphics (Hardware) Acceleration</b> to be enabled in your browser settings to process hand tracking via MediaPipe.
          </p>
          <hr style={{ border: "0", borderTop: "1px solid #ffeeba", margin: "15px 0" }} />
          <div style={{ textAlign: "left", fontSize: "14px", margin: "0" }}>
            <strong>How to fix in Google Chrome:</strong>
            <ol style={{ paddingLeft: "20px", margin: "5px 0 0 0" }}>
              <li>Go to Chrome Settings (<code>chrome://settings/system</code>)</li>
              <li>Turn on <b>"Use graphics acceleration when available"</b></li>
              <li>Relaunch Chrome and reload this page</li>
            </ol>
          </div>
          <p style={{ marginTop: "15px", fontSize: "14px" }}>
            <em>Alternatively, you can open and run this app smoothly in <b>Safari</b>.</em>
          </p>
        </div>
      </div>
    );
  }

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
          <button className="chart-btn" onClick={() => setShowChart(true)}>
            View ASL Alphabet Chart
          </button>
          
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

      {/* ASL Chart Modal Popup */}
      {showChart && (
        <div className="modal-overlay" onClick={() => setShowChart(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>American Sign Language (ASL) Alphabet</h3>
            <img 
              src={fingerspellingImg} 
              alt="ASL Fingerspelling Chart" 
            />
            <br />
            <button className="close-btn" onClick={() => setShowChart(false)}>
              Close Chart
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;