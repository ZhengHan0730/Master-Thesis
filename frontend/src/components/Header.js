import React from "react";
import { Link } from "react-router-dom";
import "./Header.css"; // 引入 CSS 文件

const Header = () => {
  return (
    <header className="header">
      <div className="logo">
        <img src="/uzh.png" alt="Universität Zürich Logo" />
        
      </div>

      <div className="title">PREDANO Privacy Enhancement Toolbox</div>

      <nav className="nav">
        <Link to="/home" className="nav-link">Home</Link>
        <Link to="/demo" className="nav-link">Demo</Link>
        <Link to="/application" className="nav-link">Application</Link>
        <Link to="/evaluate" className="nav-link highlight">Evaluation</Link> {/* 新增导航 */}
        <Link to="/contact" className="nav-link">Contact</Link>
      </nav>
    </header>
  );
};

export default Header;
