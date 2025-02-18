import {
  Layout,
  Form,
  Select,
  InputNumber,
  Button,
  Typography,
  message,
  Tooltip,
} from "antd";
import AppHeader from "./Header";
import React, { useState } from "react";
import { useLocation } from "react-router-dom";
const { Option } = Select;
const { Title } = Typography;
const { Content } = Layout;

const AnonymityForm = ({ file }) => {
  const [algorithm, setAlgorithm] = useState("k-anonymity");
  const location = useLocation();
  const { identifier, quasiIdentifiers, sensitiveColumn, hierarchyRules, csvHeaders } =
    location.state || {
      identifier: "",
      quasiIdentifiers: [],
      file: null,
      sensitiveColumn: "",
      hierarchyRules: {},
      csvHeaders: []
    };

  const onAlgorithmChange = (value) => {
    setAlgorithm(value);
  };

  const downloadCSV = (data, filename) => {
    const blob = new Blob([data], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const onFinish = async (values) => {
    const formData = new FormData();

    formData.append("file", file);
    formData.append("sensitive_column", sensitiveColumn);
    formData.append("identifier", identifier);
    formData.append("privacy_model", values.algorithm);
    formData.append("k", values.kValue || "");
    formData.append("l", values.lValue || "");
    formData.append("t", values.tValue || "");
    formData.append("m", values.mValue || "");
    formData.append("delta_min", values.deltaMin || "");
    formData.append("delta_max", values.deltaMax || "");
    formData.append("beta", values.betaValue || ""); // add β-value
    formData.append("delta_disclosure", values.deltaDisclosure || ""); // Add delta-disclosure value
    formData.append("p", values.pValue || ""); // Add p-value
    formData.append("suppression_threshold", values.supRate || "");
    formData.append("quasi_identifiers", quasiIdentifiers.join(","));
    formData.append("hierarchy_rules", JSON.stringify(hierarchyRules));
    formData.append("c", values.cValue || "");

    try {
      const response = await fetch("/anonymize", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log("Anonymized Data:", result);

        const header = Object.keys(result[0]).join(",") + "\n";
        const rows = result
          .map((row) => Object.values(row).join(","))
          .join("\n");
        const csvData = header + rows;

        downloadCSV(csvData, "anonymized_data.csv");
        message.success("Data anonymization successful! CSV downloaded.");
      } else {
        const error = await response.json();
        console.error(`Server Error: ${error.error}`);
        message.error(`Error: ${error.error}`);
      }
    } catch (error) {
      console.error("Network Error during data anonymization:", error);
      message.error(
        "A network error occurred while processing the data. Please check your connection and try again."
      );
    }
  };

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <AppHeader />
      <Content style={{ maxWidth: 600, margin: "0 auto", padding: "20px" }}>
        <Title level={3}>Algorithm Selection</Title>
        <Form
          layout="vertical"
          onFinish={onFinish}
          style={{ marginTop: "20px" }}
        >
          <Form.Item
            name="algorithm"
            label="Algorithm"
            rules={[{ required: true, message: "Please select an algorithm!" }]}
          >
            <Select placeholder="Select Algorithm" onChange={onAlgorithmChange}>
              <Option value="k-anonymity">K-anonymity</Option>
              <Option value="l-diversity">L-diversity</Option>
              <Option value="t-closeness">T-closeness</Option>
              <Option value="km-anonymity">Km-anonymity</Option>
              <Option value="delta-presence">δ-Presence</Option>
              <Option value="beta-likeness">β-Likeness</Option> {/* 添加 β-likeness */}
              <Option value="delta-disclosure">δ-Disclosure</Option> {/* Add new option */}
              <Option value="p-sensitivity">p-Sensitivity</Option>
              <Option value="ck-safety">(c,k)-Safety</Option>
            </Select>
          </Form.Item>

          {(algorithm === "k-anonymity" ||
            algorithm === "l-diversity" ||
            algorithm === "t-closeness" ||
            algorithm === "km-anonymity" ||
            algorithm === "ck-safety") && (
              <Tooltip title="The k value in k-anonymity ensures that each record is indistinguishable from at least k-1 other records, protecting individuals from being uniquely identified.">
                <Form.Item
                  name="kValue"
                  label="K-value"
                  rules={[{ required: true, message: "Please input the K-value!" }]}
                >
                  <InputNumber
                    min={1}
                    style={{ width: "100%" }}
                    placeholder="Enter K-value"
                  />
                </Form.Item>
              </Tooltip>
          )}

          {algorithm === "l-diversity" && (
            <Tooltip title="The l value in l-diversity ensures that each equivalence class has at least l distinct sensitive values to protect against attribute disclosure.">
              <Form.Item
                name="lValue"
                label="L-value"
                rules={[
                  {
                    required: algorithm === "l-diversity",
                    message: "Please input the L-value!",
                  },
                ]}
              >
                <InputNumber
                  min={1}
                  style={{ width: "100%" }}
                  placeholder="Enter L-value"
                />
              </Form.Item>
            </Tooltip>
          )}

          {algorithm === "t-closeness" && (
            <Tooltip title="The t value in t-closeness ensures that the distribution of sensitive attributes in each group is within a t threshold of the overall distribution, limiting information leakage.">
              <Form.Item
                name="tValue"
                label="T-value"
                rules={[
                  {
                    required: algorithm === "t-closeness",
                    message: "Please input the T-value!",
                  },
                ]}
              >
                <InputNumber
                  min={0.01}
                  max={1}
                  step={0.01}
                  style={{ width: "100%" }}
                  placeholder="Enter T-value"
                />
              </Form.Item>
            </Tooltip>
          )}

          {algorithm === "km-anonymity" && (
            <Tooltip title="The m value in km-anonymity specifies the minimum number of distinct sensitive attribute values within each equivalence class to prevent sensitive attribute disclosure.">
              <Form.Item
                name="mValue"
                label="M-value"
                rules={[
                  {
                    required: algorithm === "km-anonymity",
                    message: "Please input the M-value!",
                  },
                ]}
              >
                <InputNumber
                  min={1}
                  style={{ width: "100%" }}
                  placeholder="Enter M-value"
                />
              </Form.Item>
            </Tooltip>
          )}

          {algorithm === "delta-presence" && (
            <>
              <Tooltip title="The minimum threshold for the proportion of sensitive values in each equivalence class">
                <Form.Item
                  name="deltaMin"
                  label="δ-min value"
                  rules={[
                    {
                      required: algorithm === "delta-presence",
                      message: "Please input the δ-min value!",
                    },
                  ]}
                >
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.01}
                    style={{ width: "100%" }}
                    placeholder="Enter δ-min value"
                  />
                </Form.Item>
              </Tooltip>
              <Tooltip title="The maximum threshold for the proportion of sensitive values in each equivalence class">
                <Form.Item
                  name="deltaMax"
                  label="δ-max value"
                  rules={[
                    {
                      required: algorithm === "delta-presence",
                      message: "Please input the δ-max value!",
                    },
                  ]}
                >
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.01}
                    style={{ width: "100%" }}
                    placeholder="Enter δ-max value"
                  />
                </Form.Item>
              </Tooltip>
            </>
          )}

          {algorithm === "beta-likeness" && ( // 添加 β-likeness 的表单项
            <Tooltip title="The β value specifies the maximum allowed difference between the sensitive attribute distribution in equivalence classes and the global distribution.">
              <Form.Item
                name="betaValue"
                label="β-value"
                rules={[
                  {
                    required: algorithm === "beta-likeness",
                    message: "Please input the β-value!",
                  },
                ]}
              >
                <InputNumber
                  min={0.01}
                  max={1}
                  step={0.01}
                  style={{ width: "100%" }}
                  placeholder="Enter β-value"
                />
              </Form.Item>
            </Tooltip>
          )}
           {algorithm === "delta-disclosure" && (
            <Tooltip title="The δ value in δ-disclosure privacy ensures that the probability of inferring sensitive values remains below the specified threshold.">
              <Form.Item
                name="deltaDisclosure"
                label="δ-value"
                rules={[
                  {
                    required: algorithm === "delta-disclosure",
                    message: "Please input the δ-value!",
                  },
                ]}
              >
                <InputNumber
                  min={0}
                  max={1}
                  step={0.01}
                  style={{ width: "100%" }}
                  placeholder="Enter δ-value"
                />
              </Form.Item>
            </Tooltip>
          )}

            {algorithm === "p-sensitivity" && (
              <Tooltip title="The p value in p-sensitivity ensures that each equivalence class contains at least p different values for the sensitive attribute, providing stronger protection against attribute disclosure.">
                <Form.Item
                  name="pValue"
                  label="p-value"
                  rules={[
                    {
                      required: algorithm === "p-sensitivity",
                      message: "Please input the p-value!",
                    },
                  ]}
                >
                  <InputNumber
                    min={1}
                    style={{ width: "100%" }}
                    placeholder="Enter p-value"
                  />
                </Form.Item>
              </Tooltip>
            )}

            {algorithm === "ck-safety" && (
              <Tooltip title="The c value specifies the maximum confidence (in percentage) allowed for inferring any sensitive value within an equivalence class.">
                <Form.Item
                  name="cValue"
                  label="C-value (%)"
                  rules={[
                    {
                      required: true,
                      message: "Please input the C-value!",
                    },
                  ]}
                >
                  <InputNumber
                    min={0}
                    max={100}
                    step={1}
                    style={{ width: "100%" }}
                    placeholder="Enter C-value (e.g., 60 for 60%)"
                  />
                </Form.Item>
              </Tooltip>
            )}

          <Tooltip title="Suppression Rate refers to the percentage of data values removed or hidden during anonymization to meet privacy requirements.">
            <Form.Item
              name="supRate"
              label="Suppression Rate"
              rules={[
                { required: true, message: "Please input the suppression rate!" },
              ]}
            >
              <InputNumber
                min={0}
                max={1}
                step={0.01}
                style={{ width: "100%" }}
                placeholder="Enter suppression rate"
              />
            </Form.Item>
          </Tooltip>

          <Form.Item>
            <Button
              htmlType="submit"
              style={{
                width: '100%',
                color: 'white',
                backgroundColor: 'black',
                transition: 'background-color 0.3s, color 0.3s',
              }}
              block
            >
              Submit
            </Button>
          </Form.Item>
        </Form>
      </Content>
    </Layout>
  );
};

export default AnonymityForm;
