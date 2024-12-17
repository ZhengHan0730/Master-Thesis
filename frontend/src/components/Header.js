import React from 'react';
import './Header.css'; // 引入CSS文件
const Header = () => {
  return (
    <header className="header">
      <div className="logo">
        <img src="/uzh.png" alt="Universität Zürich Logo" />
        {/* <span>Universität Zürich UZH</span> */}
      </div>
      <div className="title">
        PREDANO Privacy Enhancement Toolbox
      </div>
      <nav className="nav">
        <a href="/home" className="nav-link">Home</a>
        <a href="/demo" className="nav-link">Demo</a>
        <a href="/application" className="nav-link">Application</a>
        <a href="/contact" className="nav-link">Contact</a>
      </nav>
      {/* <div className="actions">
        <button className="sign-in">Sign in</button>
        <button className="register">Register</button>
      </div> */}
    </header>
  );
};

export default Header;