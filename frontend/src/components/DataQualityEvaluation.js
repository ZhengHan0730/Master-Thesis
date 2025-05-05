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
  Space,
  Divider,
  Alert,
  Steps
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
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import AppHeader from "./Header";
import "./DataQualityEvaluation.css";

const { Title } = Typography;
const { Content } = Layout;
const { TabPane } = Tabs;
const { Step } = Steps;

// 方法分类
const numericMethods = ["mean", "median", "variance", "wasserstein", "ks_similarity", "pearson", "spearman"];
const textMethods = ["js-divergence", "mutual-information"];
const mlMethods = ["random-forest", "svm", "mlp"];
const unsupervisedMethods = ["unsupervised-quality"];

const DataQualityEvaluation = () => {
  // 基本文件上传状态
  const [originalFile, setOriginalFile] = useState(null);
  const [anonymizedFile, setAnonymizedFile] = useState(null);
  
  // ML文件上传状态
  const [originalTrainFile, setOriginalTrainFile] = useState(null);
  const [originalTestFile, setOriginalTestFile] = useState(null);
  const [anonymizedTrainFile, setAnonymizedTrainFile] = useState(null);
  const [anonymizedTestFile, setAnonymizedTestFile] = useState(null);
  
  const [columnOptions, setColumnOptions] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [selectedStatisticalMethods, setSelectedStatisticalMethods] = useState([]);
  const [labelColumn, setLabelColumn] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState([]);
  const [resultId, setResultId] = useState(null);
  
  // 添加步骤控制
  const [currentStep, setCurrentStep] = useState(0);

  // 检查是否包含ML方法
  const isMLMethod = selectedStatisticalMethods.some((m) => mlMethods.includes(m));
  const isTextMethod = selectedStatisticalMethods.some((m) => textMethods.includes(m));
  const isUnsupervisedMethod = selectedStatisticalMethods.some((m) => unsupervisedMethods.includes(m));

  // 处理常规文件上传
  const handleFileChange = (info, type) => {
    const fileList = info.fileList;
    const fileObj = fileList[fileList.length - 1]?.originFileObj;

    if (!fileObj) {
      message.error("上传失败，文件未正确获取！");
      return;
    }

    if (type === "original") {
      setOriginalFile(fileObj);
      readColumns(fileObj);
    } else if (type === "anonymized") {
      setAnonymizedFile(fileObj);
    }

    message.success(`${info.file.name} uploaded successfully`);
  };

  // 处理ML数据集文件上传
  const handleMLFileChange = (info, type) => {
    const fileList = info.fileList;
    const fileObj = fileList[fileList.length - 1]?.originFileObj;

    if (!fileObj) {
      message.error("上传失败，文件未正确获取！");
      return;
    }

    switch (type) {
      case "original_train":
        setOriginalTrainFile(fileObj);
        readColumns(fileObj);
        break;
      case "original_test":
        setOriginalTestFile(fileObj);
        break;
      case "anonymized_train":
        setAnonymizedTrainFile(fileObj);
        break;
      case "anonymized_test":
        setAnonymizedTestFile(fileObj);
        break;
      default:
        break;
    }

    message.success(`${info.file.name} uploaded successfully`);
  };

  // 从文件中读取列名
  const readColumns = (fileObj) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const firstLine = text.split("\n")[0];
      const columns = firstLine.split(/,|\t/).map((c) => c.trim());
      setColumnOptions(columns);
    };
    reader.readAsText(fileObj);
  };

  // 判断当前步骤是否可以继续
  const canProceedToNextStep = () => {
    if (currentStep === 0) {
      // 第一步：至少选择一个评估方法
      return selectedStatisticalMethods.length > 0;
    }
    
    if (currentStep === 1) {
      // 第二步：确保上传了正确的文件
      if (isMLMethod) {
        return originalTrainFile && originalTestFile && anonymizedTrainFile && anonymizedTestFile;
      } else {
        return originalFile && anonymizedFile;
      }
    }
    
    if (currentStep === 2) {
      // 第三步：选择列和标签列（如果需要）
      if (selectedColumns.length === 0) return false;
      if (isMLMethod && !labelColumn) return false;
      return true;
    }

    if (currentStep === 3) {
      // 第四步：确认页面，没有特殊验证要求，所以返回true
      return true;
    }
    
    return false;
  };

  // 处理步骤变化
  const handleNextStep = () => {
    if (canProceedToNextStep()) {
      setCurrentStep(currentStep + 1);
    } else {
      if (currentStep === 0) {
        message.error("请至少选择一种评估方法！");
      } else if (currentStep === 1) {
        if (isMLMethod) {
          message.error("请上传所有四个数据集文件！");
        } else {
          message.error("请上传原始数据和匿名化数据文件！");
        }
      } else if (currentStep === 2) {
        if (selectedColumns.length === 0) {
          message.error("请选择至少一列进行评估！");
        } else if (isMLMethod && !labelColumn) {
          message.error("请选择一个标签列用于机器学习评估！");
        }
      }
    }
  };

  const handlePrevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  const handleEvaluate = async () => {
    if (!canProceedToNextStep()) {
      message.error("请完成所有必要的设置！");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    
    // 根据选择的评估方法添加不同的文件
    if (isMLMethod) {
      formData.append("original_train_file", originalTrainFile);
      formData.append("original_test_file", originalTestFile);
      formData.append("anonymized_train_file", anonymizedTrainFile);
      formData.append("anonymized_test_file", anonymizedTestFile);
    } else {
      formData.append("original_file", originalFile);
      formData.append("anonymized_file", anonymizedFile);
    }
    
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
      
      // 评估完成后前进到结果页
      setCurrentStep(4);
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

  const getTableColumns = () => {
    if (isMLMethod) {
      return [
        { title: "Metric", dataIndex: "metric", key: "metric" },
        { title: "Dataset", dataIndex: "dataset", key: "dataset" },
        { title: "Accuracy", dataIndex: "accuracy", key: "accuracy" },
        { title: "F1 Score", dataIndex: "f1_score", key: "f1_score" },
        { title: "Precision", dataIndex: "precision", key: "precision" },
      ];
    } else if (isTextMethod) {
      return [
        { title: "Column", dataIndex: "column", key: "column" },
        { title: "Metric", dataIndex: "metric", key: "metric" },
        { title: "Difference", dataIndex: "difference", key: "difference" },
      ];
    } else if (isUnsupervisedMethod) {
      return [
        { title: "Metric", dataIndex: "metric", key: "metric" },
        { title: "Original", dataIndex: "original", key: "original" },
        { title: "Anonymized", dataIndex: "anonymized", key: "anonymized" },
        { title: "Difference", dataIndex: "difference", key: "difference" },
        { title: "Error", dataIndex: "error", key: "error" },
      ]; 
    } else {
      return [
        { title: "Column", dataIndex: "column", key: "column" },
        { title: "Metric", dataIndex: "metric", key: "metric" },
        { title: "Original", dataIndex: "original", key: "original" },
        { title: "Anonymized", dataIndex: "anonymized", key: "anonymized" },
        { title: "Difference", dataIndex: "difference", key: "difference" },
        { title: "Error", dataIndex: "error", key: "error" },
      ];
    }
  };

  // Filter results for specific ML methods
  const getRFResults = () => resultData.filter((r) => r.metric === "random-forest");
  const getSVMResults = () => resultData.filter((r) => r.metric === "svm");
  const getMLPResults = () => resultData.filter((r) => r.metric === "mlp");
  
  // Get unsupervised methods results
  const getUnsupervisedResults = () => resultData.filter((r) => 
    ["kmeans_silhouette", "knn_neighbor_preservation", "local_outlier_factor"].includes(r.metric)
  );

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

  // Prepare data for unsupervised radar chart
  const prepareUnsupervisedRadarData = () => {
    const unsupervisedResults = getUnsupervisedResults();
    if (unsupervisedResults.length === 0) return [];
    
    // Format radar data with both original and anonymized values
    return [
      {
        subject: "KMeans Silhouette",
        original: unsupervisedResults.find(r => r.metric === "kmeans_silhouette")?.original || 0,
        anonymized: unsupervisedResults.find(r => r.metric === "kmeans_silhouette")?.anonymized || 0,
      },
      {
        subject: "KNN Preservation",
        original: unsupervisedResults.find(r => r.metric === "knn_neighbor_preservation")?.original || 0,
        anonymized: unsupervisedResults.find(r => r.metric === "knn_neighbor_preservation")?.anonymized || 0,
      },
      {
        subject: "Local Outlier Factor",
        original: unsupervisedResults.find(r => r.metric === "local_outlier_factor")?.original || 0,
        anonymized: unsupervisedResults.find(r => r.metric === "local_outlier_factor")?.anonymized || 0,
      }
    ];
  };

  // 渲染步骤内容
  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        // 步骤1：选择评估方法
        return (
          <>
            <Title level={4}>Step 1: Select Evaluation Methods</Title>
            <Form layout="vertical">
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

              <Form.Item label="Unsupervised Learning Evaluation Methods">
                <Checkbox.Group
                  options={unsupervisedMethods.map((m) => ({ label: m, value: m }))}
                  value={selectedStatisticalMethods.filter(m => unsupervisedMethods.includes(m))}
                  onChange={(vals) => {
                    const currentNonUnsupervised = selectedStatisticalMethods.filter(m => !unsupervisedMethods.includes(m));
                    setSelectedStatisticalMethods([...vals, ...currentNonUnsupervised]);
                  }}
                />
              </Form.Item>
            </Form>
          </>
        );
        
      case 1:
        // 步骤2：上传文件
        return (
          <>
            <Title level={4}>Step 2: Upload Data Files</Title>
            {isMLMethod ? (
              // ML评估需要四个文件
              <>
                <Alert
                  message="ML Evaluation Mode"
                  description="Machine learning evaluation requires uploading four datasets: original training set, original test set, anonymized training set, and anonymized test set."
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
                
                <Divider orientation="left">Training Sets</Divider>
                <Form.Item label="Upload Original Training Data (CSV/TSV)">
                  <Upload 
                    beforeUpload={() => false} 
                    onChange={(info) => handleMLFileChange(info, "original_train")}
                    accept=".csv,.tsv" 
                    showUploadList={true}
                  >
                    <Button icon={<UploadOutlined />}>Upload Original Training Data</Button>
                  </Upload>
                </Form.Item>

                <Form.Item label="Upload Anonymized Training Data (CSV/TSV)">
                  <Upload 
                    beforeUpload={() => false} 
                    onChange={(info) => handleMLFileChange(info, "anonymized_train")} 
                    accept=".csv,.tsv" 
                    showUploadList={true}
                  >
                    <Button icon={<UploadOutlined />}>Upload Anonymized Training Data</Button>
                  </Upload>
                </Form.Item>
                
                <Divider orientation="left">Test Sets</Divider>
                <Form.Item label="Upload Original Test Data (CSV/TSV)">
                  <Upload 
                    beforeUpload={() => false} 
                    onChange={(info) => handleMLFileChange(info, "original_test")}
                    accept=".csv,.tsv" 
                    showUploadList={true}
                  >
                    <Button icon={<UploadOutlined />}>Upload Original Test Data</Button>
                  </Upload>
                </Form.Item>

                <Form.Item label="Upload Anonymized Test Data (CSV/TSV)">
                  <Upload 
                    beforeUpload={() => false} 
                    onChange={(info) => handleMLFileChange(info, "anonymized_test")} 
                    accept=".csv,.tsv" 
                    showUploadList={true}
                  >
                    <Button icon={<UploadOutlined />}>Upload Anonymized Test Data</Button>
                  </Upload>
                </Form.Item>
              </>
            ) : (
              // 常规评估只需要两个文件
              <>
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
              </>
            )}
          </>
        );
        
      case 2:
        // 步骤3：选择列
        return (
          <>
            <Title level={4}>Step 3: Select Columns for Evaluation</Title>
            
            <Form layout="vertical">
              <Form.Item label="Select Columns for Evaluation">
                <Checkbox.Group options={columnOptions} value={selectedColumns} onChange={setSelectedColumns} />
              </Form.Item>

              {isMLMethod && (
                <Form.Item label="Select Label Column for Machine Learning">
                  <Checkbox.Group
                    options={columnOptions}
                    value={labelColumn ? [labelColumn] : []}
                    onChange={(vals) => setLabelColumn(vals[0] || null)}
                  />
                </Form.Item>
              )}
            </Form>
          </>
        );
        
      case 3:
        // 步骤4：确认和提交
        return (
          <>
            <Title level={4}>Step 4: Review and Submit</Title>
            
            <Card title="Evaluation Summary" bordered={false} style={{ marginBottom: 20 }}>
              <p><strong>Selected Evaluation Methods:</strong> {selectedStatisticalMethods.join(", ")}</p>
              <p><strong>Selected Columns:</strong> {selectedColumns.join(", ")}</p>
              {isMLMethod && <p><strong>Label Column:</strong> {labelColumn}</p>}
              <p><strong>Files Uploaded:</strong> {isMLMethod ? 
                "Original Train, Original Test, Anonymized Train, Anonymized Test" : 
                "Original Data, Anonymized Data"}
              </p>
            </Card>
            
            <Alert
              message="Ready to Evaluate"
              description="Review your settings above. If everything looks correct, click the Submit Evaluation button to start the evaluation process."
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
          </>
        );
        
      case 4:
        // 步骤5：查看结果
        return (
          <>
            <Title level={4}>Evaluation Results (ID: {resultId})</Title>
            
            <Button 
              icon={<DownloadOutlined />} 
              onClick={handleDownloadCSV} 
              style={{ marginBottom: 16 }}
              disabled={!resultId}
            >
              Download CSV
            </Button>

            {resultData.length > 0 ? (
              <>
                <Table 
                  dataSource={resultData.map((item, index) => ({ key: index, ...item }))} 
                  pagination={false} 
                  bordered 
                  columns={getTableColumns()} 
                />

                {/* ML model evaluation charts */}
                {isMLMethod && (
                  <div style={{ marginTop: 40 }}>
                    <Title level={4}>Machine Learning Model Evaluation</Title>
                    
                    <Tabs defaultActiveKey="1">
                    {selectedStatisticalMethods.includes("random-forest") && getRFResults().length > 0 && (
                      <TabPane tab="Random Forest" key="1">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={[
                              {
                                metric: "Accuracy",
                                Original: getRFResults().find(r => r.dataset === "Original")?.accuracy || 0,
                                Anonymized: getRFResults().find(r => r.dataset === "Anonymized")?.accuracy || 0
                              },
                              {
                                metric: "F1 Score",
                                Original: getRFResults().find(r => r.dataset === "Original")?.f1_score || 0,
                                Anonymized: getRFResults().find(r => r.dataset === "Anonymized")?.f1_score || 0
                              },
                              {
                                metric: "Precision",
                                Original: getRFResults().find(r => r.dataset === "Original")?.precision || 0,
                                Anonymized: getRFResults().find(r => r.dataset === "Anonymized")?.precision || 0
                              }
                            ]}
                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="metric" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="Original" name="Original" fill="#8884d8">
                              <LabelList dataKey="Original" position="top" />
                            </Bar>
                            <Bar dataKey="Anonymized" name="Anonymized" fill="#82ca9d">
                              <LabelList dataKey="Anonymized" position="top" />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </TabPane>
                    )}

                    {selectedStatisticalMethods.includes("svm") && getSVMResults().length > 0 && (
                      <TabPane tab="SVM" key="2">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={[
                              {
                                metric: "Accuracy",
                                Original: getSVMResults().find(r => r.dataset === "Original")?.accuracy || 0,
                                Anonymized: getSVMResults().find(r => r.dataset === "Anonymized")?.accuracy || 0
                              },
                              {
                                metric: "F1 Score",
                                Original: getSVMResults().find(r => r.dataset === "Original")?.f1_score || 0,
                                Anonymized: getSVMResults().find(r => r.dataset === "Anonymized")?.f1_score || 0
                              },
                              {
                                metric: "Precision",
                                Original: getSVMResults().find(r => r.dataset === "Original")?.precision || 0,
                                Anonymized: getSVMResults().find(r => r.dataset === "Anonymized")?.precision || 0
                              }
                            ]}
                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="metric" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="Original" name="Original" fill="#8884d8">
                              <LabelList dataKey="Original" position="top" />
                            </Bar>
                            <Bar dataKey="Anonymized" name="Anonymized" fill="#82ca9d">
                              <LabelList dataKey="Anonymized" position="top" />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </TabPane>
                    )}

                    {selectedStatisticalMethods.includes("mlp") && getMLPResults().length > 0 && (
                      <TabPane tab="MLP" key="3">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={[
                              {
                                metric: "Accuracy",
                                Original: getMLPResults().find(r => r.dataset === "Original")?.accuracy || 0,
                                Anonymized: getMLPResults().find(r => r.dataset === "Anonymized")?.accuracy || 0
                              },
                              {
                                metric: "F1 Score",
                                Original: getMLPResults().find(r => r.dataset === "Original")?.f1_score || 0,
                                Anonymized: getMLPResults().find(r => r.dataset === "Anonymized")?.f1_score || 0
                              },
                              {
                                metric: "Precision",
                                Original: getMLPResults().find(r => r.dataset === "Original")?.precision || 0,
                                Anonymized: getMLPResults().find(r => r.dataset === "Anonymized")?.precision || 0
                              }
                            ]}
                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="metric" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="Original" name="Original" fill="#8884d8">
                              <LabelList dataKey="Original" position="top" />
                            </Bar>
                            <Bar dataKey="Anonymized" name="Anonymized" fill="#82ca9d">
                              <LabelList dataKey="Anonymized" position="top" />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </TabPane>
                    )}
                    </Tabs>
                  </div>
                )}

                {/* Unsupervised Learning Evaluation charts */}
                {isUnsupervisedMethod && getUnsupervisedResults().length > 0 && (
                  <div style={{ marginTop: 40 }}>
                    <Title level={4}>Unsupervised Learning Evaluation</Title>
                    
                    <div style={{ display: "flex", justifyContent: "center", marginTop: 20 }}>
                      <ResponsiveContainer width="80%" height={400}>
                        <RadarChart 
                          cx="50%" 
                          cy="50%" 
                          outerRadius="80%" 
                          data={prepareUnsupervisedRadarData()}
                        >
                          <PolarGrid />
                          <PolarAngleAxis dataKey="subject" />
                          <PolarRadiusAxis angle={30} domain={[0, 1]} />
                          <Radar
                            name="Original Data"
                            dataKey="original"
                            stroke="#8884d8"
                            fill="#8884d8"
                            fillOpacity={0.6}
                          />
                          <Radar
                            name="Anonymized Data"
                            dataKey="anonymized"
                            stroke="#82ca9d"
                            fill="#82ca9d"
                            fillOpacity={0.6}
                          />
                          <Legend />
                          <Tooltip />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                    
                    <div style={{ marginTop: 20 }}>
                      <Title level={5}>Unsupervised Learning Metrics Explanation:</Title>
                      <ul>
                        <li><strong>KMeans Silhouette:</strong> Measures how well data points fit within their clusters. Higher values (closer to 1) indicate better-defined clusters.</li>
                        <li><strong>KNN Neighbor Preservation:</strong> Measures how well the K-nearest neighbors are preserved after anonymization. Higher values indicate better preservation of local data relationships.</li>
                        <li><strong>Local Outlier Factor:</strong> Assesses how outlier detection is affected by anonymization. Similar values between original and anonymized data indicate better preservation of anomaly patterns.</li>
                      </ul>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <Alert
                message="No Results"
                description="No evaluation results available yet. Please check if the evaluation was completed successfully."
                type="warning"
                showIcon
              />
            )}
            
            <div style={{ marginTop: 20 }}>
              <Button type="primary" onClick={() => setCurrentStep(0)}>
                Start New Evaluation
              </Button>
            </div>
          </>
        );
        
      default:
        return null;
    }
  };

  // 渲染底部按钮
  const renderStepButtons = () => {
    if (currentStep === 4) return null; // 结果页不需要导航按钮
    
    return (
      <div style={{ marginTop: 24, display: 'flex', justifyContent: 'space-between' }}>
        {currentStep > 0 && (
          <Button onClick={handlePrevStep}>
            Previous
          </Button>
        )}
        
        <div style={{ flex: 1 }}></div>
        
        {currentStep < 3 ? (
          <Button type="primary" onClick={handleNextStep}>
            Next
          </Button>
        ) : (
          <Button type="primary" onClick={handleEvaluate} loading={loading}>
            {loading ? "Evaluating..." : "Submit Evaluation"}
          </Button>
        )}
      </div>
    );
  };

  return (
    <Layout className="quality-layout">
      <AppHeader />
      <Content className="quality-content">
        <Card className="evaluation-card">
          <Title level={2}>Data Quality Evaluation</Title>
          
          {/* 步骤指示器 */}
          <Steps current={currentStep} style={{ marginBottom: 30 }}>
            <Step title="Select Methods" />
            <Step title="Upload Files" />
            <Step title="Select Columns" />
            <Step title="Review & Submit" />
            <Step title="Results" />
          </Steps>
          
          {/* 每个步骤的内容 */}
          {renderStepContent()}
          
          {/* 步骤按钮导航 */}
          {renderStepButtons()}
        </Card>
      </Content>
    </Layout>
  );
};

export default DataQualityEvaluation;