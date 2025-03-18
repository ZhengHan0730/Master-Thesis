import {
    Layout,
    Form,
    Upload,
    Button,
    Typography,
    message,
    Select,
    Card,
    Checkbox,
  } from "antd";
  import { UploadOutlined } from "@ant-design/icons";
  import AppHeader from "./Header";
  import React, { useState } from "react";
  
  const { Title } = Typography;
  const { Content } = Layout;
  const { Option } = Select;
  
  const statisticalMethods = [
    "Wasserstein Distance",
    "KS-Similarity",
    "Pearson & Spearman",
    "JS-Divergence",
    "Mutual Information",
    "Mean",
    "Median",
    "Variance",
  ];
  
  const mlModels = ["Random Forest", "SVM", "MLP"];
  const mlMetrics = ["Accuracy", "F1 Score", "Precision"];
  
  const DataQualityEvaluation = () => {
    const [originalFile, setOriginalFile] = useState(null);
    const [anonymizedFile, setAnonymizedFile] = useState(null);
    const [selectedMetric, setSelectedMetric] = useState(null);
    const [selectedStatisticalMethods, setSelectedStatisticalMethods] = useState([]);
    const [selectedMLModels, setSelectedMLModels] = useState([]);
    const [selectedMLMetrics, setSelectedMLMetrics] = useState([]);
    const [loading, setLoading] = useState(false);
  
    const handleFileChange = (info, type) => {
      if (info.file.status === "done") {
        if (type === "original") {
          setOriginalFile(info.file.originFileObj);
        } else {
          setAnonymizedFile(info.file.originFileObj);
        }
        message.success(`${info.file.name} uploaded successfully`);
      }
    };
  
    const handleEvaluate = async () => {
      if (!originalFile || !anonymizedFile || !selectedMetric) {
        message.error("请上传文件并选择评估方法！");
        return;
      }
  
      setLoading(true);
      const formData = new FormData();
      formData.append("original_file", originalFile);
      formData.append("anonymized_file", anonymizedFile);
      formData.append("metric", selectedMetric);
  
      if (selectedMetric === "statistical") {
        formData.append("statistical_methods", JSON.stringify(selectedStatisticalMethods));
      }
      if (selectedMetric === "ml_utility") {
        formData.append("ml_models", JSON.stringify(selectedMLModels));
        formData.append("ml_metrics", JSON.stringify(selectedMLMetrics));
      }
  
      try {
        const response = await fetch("http://127.0.0.1:5000/api/evaluate", {
          method: "POST",
          body: formData,
        });
  
        if (response.ok) {
          message.success("数据质量评估成功！");
        } else {
          const error = await response.json();
          message.error(`评估失败: ${error.error}`);
        }
      } catch (error) {
        message.error("网络错误，请检查连接并重试！");
      }
  
      setLoading(false);
    };
  
    return (
      <Layout style={{ minHeight: "100vh", backgroundColor: "#f8f9fa" }}>
        <AppHeader />
        <Content style={{ display: "flex", justifyContent: "center", padding: "60px 20px" }}>
          <Card className="evaluation-card">
            <Title level={1} className="evaluation-title">Data Quality Evaluation</Title>
            <Form layout="vertical">
              {/* 上传原始数据 */}
              <Form.Item label="Upload Original Data (CSV/TSV)">
                <Upload
                  beforeUpload={() => false}
                  onChange={(info) => handleFileChange(info, "original")}
                  accept=".csv,.tsv"
                  showUploadList={true}
                >
                  <Button icon={<UploadOutlined />}>Click to Upload</Button>
                </Upload>
              </Form.Item>
  
              {/* 上传匿名化数据 */}
              <Form.Item label="Upload Anonymized Data (CSV/TSV)">
                <Upload
                  beforeUpload={() => false}
                  onChange={(info) => handleFileChange(info, "anonymized")}
                  accept=".csv,.tsv"
                  showUploadList={true}
                >
                  <Button icon={<UploadOutlined />}>Click to Upload</Button>
                </Upload>
              </Form.Item>
  
              {/* 选择评估方法 */}
              <Form.Item label="Select Evaluation Method">
                <Select
                  placeholder="Choose an evaluation method"
                  onChange={(value) => setSelectedMetric(value)}
                >
                  <Option value="statistical">Statistical Similarity</Option>
                  <Option value="ml_utility">Machine Learning Utility</Option>
                </Select>
              </Form.Item>
  
              {/* 当选择 Statistical Similarity 时，显示详细方法 */}
              {selectedMetric === "statistical" && (
                <Form.Item label="Select Statistical Methods">
                  <Checkbox.Group
                    options={statisticalMethods}
                    onChange={setSelectedStatisticalMethods}
                  />
                </Form.Item>
              )}
  
              {/* 当选择 Machine Learning Utility 时，显示模型和评估指标 */}
              {selectedMetric === "ml_utility" && (
                <>
                  <Form.Item label="Select Machine Learning Models">
                    <Checkbox.Group options={mlModels} onChange={setSelectedMLModels} />
                  </Form.Item>
  
                  <Form.Item label="Select Evaluation Metrics">
                    <Checkbox.Group options={mlMetrics} onChange={setSelectedMLMetrics} />
                  </Form.Item>
                </>
              )}
  
              {/* 提交按钮 */}
              <Form.Item>
                <Button
                  onClick={handleEvaluate}
                  loading={loading}
                  block
                  className="submit-button"
                >
                  {loading ? "Evaluating..." : "Submit Evaluation"}
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Content>
      </Layout>
    );
  };
  
  export default DataQualityEvaluation;
  