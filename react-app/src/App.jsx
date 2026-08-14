import "./App.css";
import CustomWebcam from "./CustomWebcam";

function App() {
return (
    <div className="v-box">
      {/* Title on top */}
      <h2>Sign Language Translator</h2>

      {/* HBox holding Webcam and Text Field side-by-side */}
      <div className="h-box">
        <CustomWebcam />
        
        <div className="text-panel">
          <label htmlFor="translation">Translated Text:</label>
          <textarea
            id="translation"
            placeholder="Your translated sign language will appear here..."
            readOnly
          />
        </div>
      </div>
    </div>
  );
}

export default App