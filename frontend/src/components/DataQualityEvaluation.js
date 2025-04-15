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

const numericMethods = ["mean", "median", "variance", "wasserstein", "ks_similarity", "pearson", "spearman"];
const textMethods = ["js-divergence"];

const DataQualityEvaluation = () => {
  const [originalFile, setOriginalFile] = useState(null);
  const [anonymizedFile, setAnonymizedFile] = useState(null);
  const [columnOptions, setColumnOptions] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [selectedStatisticalMethods, setSelectedStatisticalMethods] = useState([]);
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

    setLoading(true);
    const formData = new FormData();
    formData.append("original_file", originalFile);
    formData.append("anonymized_file", anonymizedFile);
    formData.append("columns", selectedColumns.join(","));
    formData.append("metrics", selectedStatisticalMethods.join(","));

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

      let result;
      try {
        result = await response.json();
      } catch (err) {
        message.error("后端返回内容不是有效的 JSON");
        return;
      }

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

  const isTextMethod = selectedStatisticalMethods.some((m) => textMethods.includes(m));

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

              <Table
                dataSource={resultData.map((item, index) => ({ key: index, ...item }))}
                pagination={false}
                bordered
                columns={[
                  { title: "Column", dataIndex: "column", key: "column" },
                  { title: "Metric", dataIndex: "metric", key: "metric" },
                  {
                    title: "Original",
                    dataIndex: "original",
                    key: "original",
                    render: (val) =>
                      typeof val === "object"
                        ? Object.entries(val)
                            .map(([k, v]) => `${k}: ${v}`)
                            .join("\n")
                        : val,
                  },
                  {
                    title: "Anonymized",
                    dataIndex: "anonymized",
                    key: "anonymized",
                    render: (val) =>
                      typeof val === "object"
                        ? Object.entries(val)
                            .map(([k, v]) => `${k}: ${v}`)
                            .join("\n")
                        : val,
                  },
                  { title: "Difference", dataIndex: "difference", key: "difference" },
                  { title: "Error", dataIndex: "error", key: "error" },
                ]}
              />

              {!isTextMethod && (
                <BarChart
                  width={700}
                  height={300}
                  data={resultData.filter((r) => !r.error)}
                  style={{ marginTop: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="column" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="original" fill="#8884d8" name="Original">
                    <LabelList dataKey="metric" position="top" />
                  </Bar>
                  <Bar dataKey="anonymized" fill="#82ca9d" name="Anonymized" />
                </BarChart>
              )}
            </>
          )}
        </Card>
      </Content>
    </Layout>
  );
};

export default DataQualityEvaluation;
