import React from 'react';
import './App.css';
import ImageMaskEditor from './components/ImageMaskEditor/ImageMaskEditor';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Image Mask Editor</h1>
      </header>
      <main className="App-main">
        <ImageMaskEditor />
      </main>
    </div>
  );
}

export default App;