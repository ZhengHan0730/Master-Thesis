import os
from flask import Flask

def create_app():
    app = Flask(__name__) #创建flask应用实列
    #配置上传文件的路径指向uploads
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

    if not os.path.exists(app.config['UPLOAD_FOLDER']): #检查上传的文件是否存在
        os.makedirs(app.config['UPLOAD_FOLDER']) # 如果不存在的话就创建这个文件

    with app.app_context():
        from . import routes #导入同目录下的routes.py模块
        app.register_blueprint(routes.bp)

    return app
