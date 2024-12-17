import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DatasetProcessingForm from './components/DatasetProcessingForm';
import HierarchyForm from './components/HierarchyForm';
import AnonymityForm from './components/AnonymityForm';
import TermsAndConditions from './components/TermsAndConditions'; // 导入新的页面组件
import Home from './components/Home'; // 导入新的页面组件
import Demo from './components/Demo'; // 导入新的页面组件
import Contact from './components/Contact'; // 导入新的页面组件

function App() {
  const [file, setFile] = useState(null);  // 将文件状态提升到 App 组件
  return (
    <Router>
      <Routes>
        <Route path="/application" element={<DatasetProcessingForm setFile={setFile} file={file}/>} />
        <Route path="/" element={<Home file={file} />} />
        <Route path="/home" element={<Home file={file} />} />
        <Route path="/demo" element={<Demo  file={file} />} />
        <Route path="/contact" element={<Contact  file={file} />} />
        <Route path="/hierarchy" element={<HierarchyForm  file={file} />} />
        <Route path="/anonymity" element={<AnonymityForm   file={file}/>} />
        <Route path="/terms" element={<TermsAndConditions />} /> 
      </Routes>
    </Router>
  );
}

export default App;

