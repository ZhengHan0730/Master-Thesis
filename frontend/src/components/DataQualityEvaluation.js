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

// 方法分类
const numericMethods = ["mean", "median", "variance", "wasserstein", "ks_similarity", "pearson", "spearman"];
const textMethods = ["js-divergence", "mutual-information"];
const mlMethods = ["random-forest"];

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
                value={selectedStatisticalMethods}
                onChange={setSelectedStatisticalMethods}
              />
            </Form.Item>

            <Form.Item label="Text Evaluation Methods">
              <Checkbox.Group
                options={textMethods.map((m) => ({ label: m, value: m }))}
                value={selectedStatisticalMethods}
                onChange={setSelectedStatisticalMethods}
              />
            </Form.Item>

            <Form.Item label="Machine Learning Evaluation Methods">
              <Checkbox.Group
                options={mlMethods.map((m) => ({ label: m, value: m }))}
                value={selectedStatisticalMethods}
                onChange={setSelectedStatisticalMethods}
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

              {/* random-forest图表 */}
              {isMLMethod && (
                <>
                  <Title level={5} style={{ marginTop: 40 }}>
                    Random Forest Evaluation Charts
                  </Title>

                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={resultData.filter((r) => r.dataset)}
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
                </>
              )}
            </>
          )}
        </Card>
      </Content>
    </Layout>
  );
};

export default DataQualityEvaluation;
