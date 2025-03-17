import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

def create_app():
    """创建 Flask 应用实例"""
    app = Flask(__name__)

    # 配置上传文件目录
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

    # 确保上传目录存在
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # 允许 CORS，确保前端 http://localhost:3000 访问 API
    CORS(app, origins="http://localhost:3000", supports_credentials=True)

    # 设置日志记录
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    app.logger = logger

    with app.app_context():
        # 导入并注册 Blueprint
        from . import routes
        app.register_blueprint(routes.bp, url_prefix='/api')

    @app.before_request
    def log_request_info():
        """在每个请求前打印请求信息，方便调试"""
        app.logger.info(f"Request Headers: {dict(request.headers)}")
        app.logger.info(f"Request Method: {request.method}")
        app.logger.info(f"Request Path: {request.path}")

    return app
