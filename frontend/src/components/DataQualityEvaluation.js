import React, { useState } from "react";
import {
  Layout,
  Form,
  Upload,
  Button,
  Typography,
  message,
  Card,
  Checkbox,
  Table,
  Tabs,
} from "antd";
import { UploadOutlined, DownloadOutlined } from "@ant-design/icons";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LabelList,
  ResponsiveContainer,
} from "recharts";
import AppHeader from "./Header";
import "./DataQualityEvaluation.css";

const { Title } = Typography;
const { Content } = Layout;
const { TabPane } = Tabs;

// 方法分类
const numericMethods = ["mean", "median", "variance", "wasserstein", "ks_similarity", "pearson", "spearman"];
const textMethods = ["js-divergence", "mutual-information"];
const mlMethods = ["random-forest", "svm", "mlp"];

const DataQualityEvaluation = () => {
  const [originalFile, setOriginalFile] = useState(null);
  const [anonymizedFile, setAnonymizedFile] = useState(null);
  const [columnOptions, setColumnOptions] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [selectedStatisticalMethods, setSelectedStatisticalMethods] = useState([]);
  const [labelColumn, setLabelColumn] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState([]);
  const [resultId, setResultId] = useState(null);

  const handleFileChange = (info, type) => {
    const fileList = info.fileList;
    const fileObj = fileList[fileList.length - 1]?.originFileObj;

    if (!fileObj) {
      message.error("上传失败，文件未正确获取！");
      return;
    }

    if (type === "original") {
      setOriginalFile(fileObj);

      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const firstLine = text.split("\n")[0];
        const columns = firstLine.split(/,|\t/).map((c) => c.trim());
        setColumnOptions(columns);
      };
      reader.readAsText(fileObj);
    } else if (type === "anonymized") {
      setAnonymizedFile(fileObj);
    }

    message.success(`${info.file.name} uploaded successfully`);
  };

  const handleEvaluate = async () => {
    if (!originalFile || !anonymizedFile || selectedStatisticalMethods.length === 0 || selectedColumns.length === 0) {
      message.error("请上传两个文件并选择列和评估方法！");
      return;
    }

    if (selectedStatisticalMethods.some((m) => mlMethods.includes(m)) && !labelColumn) {
      message.error("请选择一个 Label 列用于机器学习评估！");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("original_file", originalFile);
    formData.append("anonymized_file", anonymizedFile);
    formData.append("columns", selectedColumns.join(","));
    formData.append("metrics", selectedStatisticalMethods.join(","));
    if (labelColumn) {
      formData.append("label", labelColumn);
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/api/evaluation", {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      if (!response.ok) {
        const error = await response.json();
        message.error(`评估失败: ${error.error}`);
        return;
      }

      const result = await response.json();
      setResultData(result.summary || []);
      setResultId(result.result_id);
      message.success("数据质量评估成功！");
    } catch (error) {
      console.error("Network error:", error);
      message.error("网络错误，请检查连接并重试！");
    }

    setLoading(false);
  };

  const handleDownloadCSV = () => {
    if (!resultId) return;
    const url = `http://127.0.0.1:5000/api/quality/result/${resultId}/download`;
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `quality_result_${resultId}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const isMLMethod = selectedStatisticalMethods.some((m) => mlMethods.includes(m));
  const isTextMethod = selectedStatisticalMethods.some((m) => textMethods.includes(m));

  const tableColumns = isMLMethod
    ? [
        { title: "Metric", dataIndex: "metric", key: "metric" },
        { title: "Dataset", dataIndex: "dataset", key: "dataset" },
        { title: "Accuracy", dataIndex: "accuracy", key: "accuracy" },
        { title: "F1 Score", dataIndex: "f1_score", key: "f1_score" },
        { title: "Precision", dataIndex: "precision", key: "precision" },
      ]
    : isTextMethod
    ? [
        { title: "Column", dataIndex: "column", key: "column" },
        { title: "Metric", dataIndex: "metric", key: "metric" },
        { title: "Difference", dataIndex: "difference", key: "difference" },
      ]
    : [
        { title: "Column", dataIndex: "column", key: "column" },
        { title: "Metric", dataIndex: "metric", key: "metric" },
        { title: "Original", dataIndex: "original", key: "original" },
        { title: "Anonymized", dataIndex: "anonymized", key: "anonymized" },
        { title: "Difference", dataIndex: "difference", key: "difference" },
        { title: "Error", dataIndex: "error", key: "error" },
      ];

  // Filter results for specific ML methods
  const getRFResults = () => resultData.filter((r) => r.metric === "random-forest");
  const getSVMResults = () => resultData.filter((r) => r.metric === "svm");
  const getMLPResults = () => resultData.filter((r) => r.metric === "mlp");

  // Get all ML methods that have been selected and have results
  const getSelectedMLMethods = () => {
    const methods = [];
    if (selectedStatisticalMethods.includes("random-forest") && getRFResults().length > 0) methods.push("random-forest");
    if (selectedStatisticalMethods.includes("svm") && getSVMResults().length > 0) methods.push("svm");
    if (selectedStatisticalMethods.includes("mlp") && getMLPResults().length > 0) methods.push("mlp");
    return methods;
  };

  // Determine if we should show the model comparison tab
  const shouldShowComparison = () => {
    const methods = getSelectedMLMethods();
    return methods.length >= 2;
  };

  return (
    <Layout className="quality-layout">
      <AppHeader />
      <Content className="quality-content">
        <Card className="evaluation-card">
          <Title level={2}>Data Quality Evaluation</Title>
          <Form layout="vertical">
            <Form.Item label="Upload Original Data (CSV/TSV)">
              <Upload beforeUpload={() => false} onChange={(info) => handleFileChange(info, "original")} accept=".csv,.tsv" showUploadList={true}>
                <Button icon={<UploadOutlined />}>Click to Upload</Button>
              </Upload>
            </Form.Item>

            <Form.Item label="Upload Anonymized Data (CSV/TSV)">
              <Upload beforeUpload={() => false} onChange={(info) => handleFileChange(info, "anonymized")} accept=".csv,.tsv" showUploadList={true}>
                <Button icon={<UploadOutlined />}>Click to Upload</Button>
              </Upload>
            </Form.Item>

            <Form.Item label="Select Columns for Evaluation">
              <Checkbox.Group options={columnOptions} value={selectedColumns} onChange={setSelectedColumns} />
            </Form.Item>

            <Form.Item label="Numeric Evaluation Methods">
              <Checkbox.Group
                options={numericMethods.map((m) => ({ label: m, value: m }))}
                value={selectedStatisticalMethods.filter(m => numericMethods.includes(m))}
                onChange={(vals) => {
                  const currentNonNumeric = selectedStatisticalMethods.filter(m => !numericMethods.includes(m));
                  setSelectedStatisticalMethods([...vals, ...currentNonNumeric]);
                }}
              />
            </Form.Item>

            <Form.Item label="Text Evaluation Methods">
              <Checkbox.Group
                options={textMethods.map((m) => ({ label: m, value: m }))}
                value={selectedStatisticalMethods.filter(m => textMethods.includes(m))}
                onChange={(vals) => {
                  const currentNonText = selectedStatisticalMethods.filter(m => !textMethods.includes(m));
                  setSelectedStatisticalMethods([...vals, ...currentNonText]);
                }}
              />
            </Form.Item>

            <Form.Item label="Machine Learning Evaluation Methods">
              <Checkbox.Group
                options={mlMethods.map((m) => ({ label: m, value: m }))}
                value={selectedStatisticalMethods.filter(m => mlMethods.includes(m))}
                onChange={(vals) => {
                  const currentNonML = selectedStatisticalMethods.filter(m => !mlMethods.includes(m));
                  setSelectedStatisticalMethods([...vals, ...currentNonML]);
                }}
              />
            </Form.Item>

            {selectedStatisticalMethods.some((m) => mlMethods.includes(m)) && (
              <Form.Item label="Select Label Column for Machine Learning">
                <Checkbox.Group
                  options={columnOptions}
                  value={labelColumn ? [labelColumn] : []}
                  onChange={(vals) => setLabelColumn(vals[0] || null)}
                />
              </Form.Item>
            )}

            <Form.Item>
              <Button type="primary" onClick={handleEvaluate} loading={loading} block>
                {loading ? "Evaluating..." : "Submit Evaluation"}
              </Button>
            </Form.Item>
          </Form>

          {resultData.length > 0 && (
            <>
              <Title level={4} style={{ marginTop: 30 }}>
                Evaluation Results (ID: {resultId})
              </Title>

              <Button icon={<DownloadOutlined />} onClick={handleDownloadCSV} style={{ marginBottom: 16 }}>
                Download CSV
              </Button>

              <Table dataSource={resultData.map((item, index) => ({ key: index, ...item }))} pagination={false} bordered columns={tableColumns} />

              {/* ML model evaluation charts */}
              {isMLMethod && (
                <div style={{ marginTop: 40 }}>
                  <Title level={4}>Machine Learning Model Evaluation</Title>
                  
                  <Tabs defaultActiveKey="1">
                    {selectedStatisticalMethods.includes("random-forest") && getRFResults().length > 0 && (
                      <TabPane tab="Random Forest" key="1">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={getRFResults()}
                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="dataset" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="accuracy" name="Accuracy" fill="#8884d8">
                              <LabelList dataKey="accuracy" position="top" />
                            </Bar>
                            <Bar dataKey="f1_score" name="F1 Score" fill="#82ca9d">
                              <LabelList dataKey="f1_score" position="top" />
                            </Bar>
                            <Bar dataKey="precision" name="Precision" fill="#ffc658">
                              <LabelList dataKey="precision" position="top" />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </TabPane>
                    )}
                    
                    {selectedStatisticalMethods.includes("svm") && getSVMResults().length > 0 && (
                      <TabPane tab="SVM" key="2">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={getSVMResults()}
                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="dataset" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="accuracy" name="Accuracy" fill="#8884d8">
                              <LabelList dataKey="accuracy" position="top" />
                            </Bar>
                            <Bar dataKey="f1_score" name="F1 Score" fill="#82ca9d">
                              <LabelList dataKey="f1_score" position="top" />
                            </Bar>
                            <Bar dataKey="precision" name="Precision" fill="#ffc658">
                              <LabelList dataKey="precision" position="top" />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </TabPane>
                    )}

                    {selectedStatisticalMethods.includes("mlp") && getMLPResults().length > 0 && (
                      <TabPane tab="MLP" key="3">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={getMLPResults()}
                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="dataset" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="accuracy" name="Accuracy" fill="#8884d8">
                              <LabelList dataKey="accuracy" position="top" />
                            </Bar>
                            <Bar dataKey="f1_score" name="F1 Score" fill="#82ca9d">
                              <LabelList dataKey="f1_score" position="top" />
                            </Bar>
                            <Bar dataKey="precision" name="Precision" fill="#ffc658">
                              <LabelList dataKey="precision" position="top" />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </TabPane>
                    )}
                    
                    {shouldShowComparison() && (
                      <TabPane tab="Model Comparison" key="4">
                        <div style={{ display: "flex", justifyContent: "space-around" }}>
                          <div style={{ width: "45%" }}>
                            <Title level={5} style={{ textAlign: "center" }}>Original Data</Title>
                            <ResponsiveContainer width="100%" height={300}>
                              <BarChart
                                data={resultData.filter(r => r.dataset === "Original" && mlMethods.includes(r.metric))}
                                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="metric" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="accuracy" name="Accuracy" fill="#8884d8">
                                  <LabelList dataKey="accuracy" position="top" />
                                </Bar>
                                <Bar dataKey="f1_score" name="F1 Score" fill="#82ca9d">
                                  <LabelList dataKey="f1_score" position="top" />
                                </Bar>
                                <Bar dataKey="precision" name="Precision" fill="#ffc658">
                                  <LabelList dataKey="precision" position="top" />
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                          
                          <div style={{ width: "45%" }}>
                            <Title level={5} style={{ textAlign: "center" }}>Anonymized Data</Title>
                            <ResponsiveContainer width="100%" height={300}>
                              <BarChart
                                data={resultData.filter(r => r.dataset === "Anonymized" && mlMethods.includes(r.metric))}
                                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="metric" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="accuracy" name="Accuracy" fill="#8884d8">
                                  <LabelList dataKey="accuracy" position="top" />
                                </Bar>
                                <Bar dataKey="f1_score" name="F1 Score" fill="#82ca9d">
                                  <LabelList dataKey="f1_score" position="top" />
                                </Bar>
                                <Bar dataKey="precision" name="Precision" fill="#ffc658">
                                  <LabelList dataKey="precision" position="top" />
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      </TabPane>
                    )}
                  </Tabs>
                </div>
              )}
            </>
          )}
        </Card>
      </Content>
    </Layout>
  );
};

export default DataQualityEvaluation;