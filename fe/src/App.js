import ImageMaskEditor from './components/ImageMaskEditor/ImageMaskEditor.jsx';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto py-8">
        <h1 className="text-3xl font-bold mb-8 text-center">Image Editor</h1>
        <ImageMaskEditor />
      </div>
    </div>
  );
}

export default App;