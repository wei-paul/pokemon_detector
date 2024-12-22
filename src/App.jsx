import { useState } from 'react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    setSelectedFile(file)
    
    // Create a preview URL for the selected image
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreviewUrl(reader.result)
      }
      reader.readAsDataURL(file)
    } else {
      setPreviewUrl(null)
    }
  }

  const handleSubmit = async () => {
    console.log('detect pokemon clicked');
    setIsProcessing(true);
    if (!selectedFile) {
      console.log('No file selected');
      return;
    }
  
    console.log('File selected:', selectedFile.name);
  
    const formData = new FormData();
    formData.append('image', selectedFile);
  
    console.log('Sending request to server');
  
    try {
      const response = await fetch('/api/detect-pokemon', {
        method: 'POST',
        body: formData,
      });
      console.log('Response received from server');
      const data = await response.json();
      console.log('Parsed response data:', data);
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="App">
      <h1>Pokemon Detector</h1>
      <label className="file-upload-label">
        Choose File
        <input type="file" onChange={handleFileChange} accept="image/*" />
      </label>
      <button onClick={handleSubmit} disabled={!selectedFile}>Detect Pokemon</button>
      
      {previewUrl && (
        <div className="preview">
          <h3>Selected Image:</h3>
          <img src={previewUrl} alt="Selected file preview" />
        </div>
      )}
      
      {result && (
        <div className="result">
          <h2>Result:</h2>
          <img src={`/api/pokemon-image/${result.similar_image_index + 1}`} alt="Most similar Pokemon" />
          <p className="confidence">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
          <p>Detected Pokemon: {result.pokemon}</p>
        </div>
      )}
      {isProcessing && (
        <div className="processing-message">
          <p>Performing magic...</p>
        </div>
      )}
    </div>
  )
}

export default App