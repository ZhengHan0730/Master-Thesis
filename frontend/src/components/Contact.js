import React from 'react';
import { Layout, Form, Input, Button, notification,Typography} from 'antd';
import AppHeader from './Header'; // 引入 Header 组件
import emailjs from 'emailjs-com'; // 引入 EmailJS 用于发送邮件

const { Content } = Layout;
const { Title} = Typography;
const Contact = () => {
  const [form] = Form.useForm();

  const handleSubmit = (values) => {
    // EmailJS 服务详情
    const serviceID = 'service_ywrua1f'; // 替换为你的 EmailJS 服务 ID
    const templateID = 'template_kqk9nfb'; // 替换为你的 EmailJS 模板 ID
    const userID = 'n1C-HVgkPsLOWyxfH'; // 替换为你的 EmailJS 用户 ID

    const templateParams = {
      name: values.name,
      surname: values.surname,
      email: values.email,
      message: values.message,
    };

    emailjs.send(serviceID, templateID, templateParams, userID)
      .then(() => {
        notification.success({
          message: 'Email Sent',
          description: 'Your message has been successfully sent!',
        });
        form.resetFields();
      })
      .catch((error) => {
        console.error('Email send failed:', error);
        notification.error({
          message: 'Email Error',
          description: 'There was an error sending your message. Please try again later.',
        });
      });
  };

  return (
    <Layout style={{ minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      <AppHeader />
      <Content style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '50px 20px' }}>
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <Title level={1}>Contact</Title>
          {/* <Text type="secondary">Context</Text> */}
        </div>
        <div style={{ width: '100%', maxWidth: '500px', backgroundColor: '#ffffff', padding: '30px', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)' }}>
          <Form form={form} layout="vertical" onFinish={handleSubmit}>
            <Form.Item
              name="name"
              label="Name"
              rules={[{ required: true, message: 'Please input your name!' }]}
            >
              <Input placeholder="Enter your name" />
            </Form.Item>
            <Form.Item
              name="surname"
              label="Surname"
              rules={[{ required: true, message: 'Please input your surname!' }]}
            >
              <Input placeholder="Enter your surname" />
            </Form.Item>
            <Form.Item
              name="email"
              label="Email"
              rules={[{ required: true, type: 'email', message: 'Please input a valid email!' }]}
            >
              <Input placeholder="Enter your email" />
            </Form.Item>
            <Form.Item
              name="message"
              label="Message"
              rules={[{ required: true, message: 'Please input your message!' }]}
            >
              <Input.TextArea placeholder="Enter your message" rows={4} />
            </Form.Item>
            <Form.Item>
            <Button 
                htmlType="submit" 
                style={{ 
                  width: '100%', 
                  color: 'white', 
                  backgroundColor: 'black',
                  transition: 'background-color 0.3s, color 0.3s',
                }}
                onMouseEnter={(e) => e.target.style.backgroundColor = 'gray'}
                onMouseLeave={(e) => e.target.style.backgroundColor = 'black'}
              >
                Submit
              </Button>
            </Form.Item>
          </Form>
        </div>
      </Content>
    </Layout>
  );
};

export default Contact;

