import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  Layout,
  Form,
  Select,
  Button,
  Typography,
  Input,
  DatePicker,
  message,
} from "antd";
import { PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import AppHeader from "./Header"; // Import the Header component
import moment from "moment";

const { Option } = Select;
const { Title, Text } = Typography;
const { Content } = Layout;
const { RangePicker } = DatePicker;

const HierarchyForm = ({ file }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { identifier, quasiIdentifiers, sensitiveColumn,csvHeaders } = location.state || {
    identifier: "",
    quasiIdentifiers: [],
    // file: null,
    sensitiveColumn: "",
    csvHeaders:[]
  };
  const [fileData, setFileData] = useState([]);
  const [selectedMethods, setSelectedMethods] = useState({});
  const [ranges, setRanges] = useState({});
  const [layers, setLayers] = useState({});
  const [maskingData, setMaskingData] = useState({}); // Store masking data
  console.log(ranges, "ra", sensitiveColumn);
  const handleFileRead = () => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      const rows = content.split("\n").map((row) => row.split(","));
      rows[0] = rows[0].map((header) => header.trim());

      // 调试信息，检查fileData是否正确加载
      console.log("File Data Loaded:", rows);

      setFileData(rows);
    };
    reader.readAsText(file);
  };

  React.useEffect(() => {
    if (file) handleFileRead();
  }, [file]);
  const getMaxMinValues = (columnName, isDate = false) => {
    // 获取列的索引
    const columnIndex = fileData[0]?.indexOf(columnName);
    console.log(columnIndex, "sss", fileData, columnName);
    if (columnIndex === -1 || columnIndex === undefined)
      return { max: null, min: null };

    // 获取列的所有值并过滤掉空值
    let columnValues = fileData
      .slice(1)
      .map((row) => row[columnIndex])
      .filter(Boolean); // 去除无效值

    if (isDate) {
      // 处理日期数据的解析
      columnValues = columnValues
        .map((value) => {
          const date = moment(
            value.trim(),
            [
              "YYYY/M/D",
              "YYYY/MM/DD",
              "M/D/YYYY",
              "MM/DD/YYYY",
              "YYYY-MM-DD",
              "DD-MM-YYYY",
            ],
            true
          );
          return date.isValid() ? date.toDate() : null;
        })
        .filter((date) => date); // 过滤无效日期
    } else {
      // 尝试将非日期值转换为数字，无法转换的保留为文本
      columnValues = columnValues.map((value) => {
        const num = parseFloat(value);
        return isNaN(num) ? value : num; // 如果无法转换为数字，则保留原始字符串
      });
    }
    let numberColumnValues =
      typeof columnValues[0] === "number" || isDate
        ? columnValues
        : columnValues.map((i) => i.match(/\d+/g)).flat();
    console.log(columnValues, numberColumnValues, "numberColumnValues");
    const filteredColumnValues = numberColumnValues.filter(
      (value) => !isNaN(value)
    );
    let numberCV = filteredColumnValues.map((i) => Number(i));
    console.log(filteredColumnValues, "filteredColumnValues");
    // 计算最大值和最小值
    const max = isDate
      ? new Date(
          columnValues
            .map((date) => date.getTime())
            .reduce((acc, time) => Math.max(acc, time), -Infinity)
        )
          .toISOString()
          .slice(0, 10) // 返回最大日期
      : numberCV.reduce((acc, val) => Math.max(acc, val), -Infinity); // 处理数字最大值

    const min = isDate
      ? new Date(
          columnValues
            .map((date) => date.getTime())
            .reduce((acc, time) => Math.min(acc, time), Infinity)
        )
          .toISOString()
          .slice(0, 10) // 返回最小日期
      : numberCV.reduce((acc, val) => Math.min(acc, val), Infinity); // 处理数字最小值
    // : typeof columnValues[0] === "number"
    // ?

    // : columnValues.sort()[0]; // 按字典序排序，获取第一个（最小）

    // const nmin = Math.min(...filteredColumnValues);

    console.log(
      columnValues,
      columnIndex,
      "columnValues",
      max,
      min,
      Math.min(...columnValues)
    );
    return { max, min };
  };

  // Generate masked string based on length
  const generateMaskedString = (length) => {
    return Array(length).fill("a");
  };

  // Handle masking toggle between 'a' and '*'
  const handleToggleMasking = (header, index) => {
    setMaskingData((prevState) => {
      const newMaskedArray = [...prevState[header]];
      newMaskedArray[index] = newMaskedArray[index] === "a" ? "*" : "a";
      return { ...prevState, [header]: newMaskedArray };
    });
  };

  const handleMethodChange = (header, method) => {
    setSelectedMethods({ ...selectedMethods, [header]: method });

    if (method === "dates") {
      const { max, min } = getMaxMinValues(header, true);
      setRanges({ ...ranges, [header]: { max, min } });
    } else if (method === "ordering") {
      console.log(111);
      const { max, min } = getMaxMinValues(header, false);
      setRanges({ ...ranges, [header]: { max, min } });
    } else if (method === "masking") {
      // 获取列索引，并计算该列中值的最小长度
      const columnIndex = fileData[0]?.indexOf(header);
      if (columnIndex === -1 || columnIndex === undefined) {
        message.error(
          `Column ${header} not found, please check if the column name is correct.`
        );
        return;
      }
      const validData = fileData
        .slice(1)
        .map((row) => row[columnIndex])
        .filter(Boolean); // 过滤掉空值或 undefined

      const minLength = Math.min(...validData.map((val) => val.length));
      console.log(
        "PATIENT column data:",
        minLength,
        fileData.map((row) => row[columnIndex])
      );
      if (minLength > 0) {
        setMaskingData({
          ...maskingData,
          [header]: generateMaskedString(minLength),
        });
      } else {
        message.error(`没有找到可遮掩的有效值：${header}`);
      }
    }
  };

  const addLayer = (header) => {
    setLayers({
      ...layers,
      [header]: [...(layers[header] || []), { min: "", max: "" }],
    });
  };

  const updateLayerValue = (header, index, key, value) => {
    const updatedLayers = [...layers[header]];
    updatedLayers[index][key] = value;
    setLayers({ ...layers, [header]: updatedLayers });
  };

  const onFinish = (values) => {
    const hierarchyRules = {};

    // Process quasi-identifiers
    quasiIdentifiers.forEach((header) => {
      hierarchyRules[header] = {
        method: values[header],
        type: "quasi-identifiers", // Add type for quasi-identifiers
      };

      // Handle different methods
      if (values[header] === "masking") {
        hierarchyRules[header].maskingString = maskingData[header].join(""); // Combine the masking data into a string
      } else if (values[header] === "dates" || values[header] === "ordering") {
        const { max, min } = ranges[header];
        hierarchyRules[header].max = max;
        hierarchyRules[header].min = min;
        hierarchyRules[header].layers = layers[header] || [];
      } else {
        hierarchyRules[header].layers = layers[header] || [];
      }
    });

    // Process sensitive attributes similarly
    hierarchyRules[sensitiveColumn] = {
      method: values[sensitiveColumn],
      type: "sensitive", // Add type for sensitive attributes
    };

    // Handle different methods for sensitive attributes
    if (values[sensitiveColumn] === "masking") {
      hierarchyRules[sensitiveColumn].maskingString =
        maskingData[sensitiveColumn].join("");
    } else if (
      values[sensitiveColumn] === "dates" ||
      values[sensitiveColumn] === "ordering"
    ) {
      const { max, min } = ranges[sensitiveColumn];
      hierarchyRules[sensitiveColumn].max = max;
      hierarchyRules[sensitiveColumn].min = min;
      hierarchyRules[sensitiveColumn].layers = layers[sensitiveColumn] || [];
    } else {
      hierarchyRules[sensitiveColumn].layers = layers[sensitiveColumn] || [];
    }

    // Navigate to the next step with updated hierarchyRules
    navigate("/anonymity", {
      state: {
        identifier,
        quasiIdentifiers,
        file,
        sensitiveColumn,
        hierarchyRules,
        csvHeaders
      },
    });
  };

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <AppHeader />
      <Content style={{ maxWidth: 600, margin: "0 auto", padding: "20px" }}>
        <Title level={3}>Hierarchy</Title>

        <Form
          layout="vertical"
          onFinish={onFinish}
          style={{ marginTop: "20px" }}
        >
          <Text type="secondary">Quasi-identifiers</Text>
          {quasiIdentifiers.map((header) => (
            <div key={header}>
              <Form.Item
                name={header}
                label={header}
                rules={[
                  {
                    required: true,
                    message: `Please select a method for ${header}!`,
                  },
                ]}
              >
                <Select
                  placeholder={`Select method for ${header}`}
                  onChange={(value) => handleMethodChange(header, value)}
                >
                  <Option value="ordering">Ordering</Option>
                  <Option value="masking">Masking</Option>
                  <Option value="category">Category</Option>
                  <Option value="dates">Dates</Option>
                </Select>
              </Form.Item>

              {/* Show range for dates/ordering methods */}
              {selectedMethods[header] &&
                (selectedMethods[header] === "dates" ||
                  selectedMethods[header] === "ordering") && (
                  <>
                    <Text type="secondary">
                      {ranges[header]
                        ? `Range: ${ranges[header].min} - ${ranges[header].max}`
                        : "Loading range..."}
                    </Text>

                    {layers[header]?.map((layer, index) => (
                      <Form.Item key={`${header}-layer-${index}`}>
                        <div style={{ marginBottom: "5px" }}>
                          <label>{`Layer ${index + 1}:`}</label>
                        </div>

                        <Input.Group
                          compact
                          style={{ display: "flex", alignItems: "center" }}
                        >
                          {selectedMethods[header] === "dates" ? (
                            <RangePicker
                              style={{ width: "85%" }}
                              onChange={(dates, dateStrings) => {
                                updateLayerValue(
                                  header,
                                  index,
                                  "min",
                                  dateStrings[0]
                                );
                                updateLayerValue(
                                  header,
                                  index,
                                  "max",
                                  dateStrings[1]
                                );
                              }}
                            />
                          ) : (
                            <>
                              <Input
                                style={{ width: "40%" }}
                                placeholder={`Min value`}
                                value={layer.min}
                                onChange={(e) =>
                                  updateLayerValue(
                                    header,
                                    index,
                                    "min",
                                    e.target.value
                                  )
                                }
                              />
                              <span style={{ margin: "0 10px" }}>-</span>
                              <Input
                                style={{ width: "40%" }}
                                placeholder={`Max value`}
                                value={layer.max}
                                onChange={(e) =>
                                  updateLayerValue(
                                    header,
                                    index,
                                    "max",
                                    e.target.value
                                  )
                                }
                              />
                            </>
                          )}
                          <Button
                            type="primary"
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() => {
                              const updatedLayers = layers[header].filter(
                                (_, layerIndex) => layerIndex !== index
                              );
                              setLayers({ ...layers, [header]: updatedLayers });
                            }}
                            style={{ marginLeft: "10px" }}
                          >
                            Delete
                          </Button>
                        </Input.Group>
                      </Form.Item>
                    ))}

                    <Form.Item>
                      <Button
                        icon={<PlusOutlined />}
                        onClick={() => addLayer(header)}
                      >
                        Add Layer
                      </Button>
                    </Form.Item>
                  </>
                )}

              {selectedMethods[header] === "masking" &&
                maskingData[header] &&
                maskingData[header].length > 0 && (
                  <div>
                    <Text>
                      Masking {maskingData[header].length} characters:
                    </Text>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(12, 1fr)", // each row 12 characters
                        gap: "10px",
                        marginTop: "10px",
                      }}
                    >
                      {maskingData[header].map((char, index) => (
                        <Button
                          key={index}
                          type={char === "*" ? "primary" : "default"}
                          onClick={() => handleToggleMasking(header, index)}
                        >
                          {char}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}
            </div>
          ))}
          {/* Sensitive attribute section */}
          <Text type="secondary">Sensitive Attribute</Text>
          <Form.Item
            name={sensitiveColumn}
            label={sensitiveColumn}
            rules={[
              {
                required: true,
                message: `Please select a method for ${sensitiveColumn}!`,
              },
            ]}
          >
            <Select
              placeholder={`Select method for ${sensitiveColumn}`}
              onChange={(value) => handleMethodChange(sensitiveColumn, value)}
            >
              <Option value="ordering">Ordering</Option>
              <Option value="masking">Masking</Option>
              <Option value="category">Category</Option>
              <Option value="dates">Dates</Option>
            </Select>
          </Form.Item>

          {/* Display operations: date range or sorting range */}
          {selectedMethods[sensitiveColumn] &&
            (selectedMethods[sensitiveColumn] === "dates" ||
              selectedMethods[sensitiveColumn] === "ordering") && (
              <>
                <Text type="secondary">
                  {ranges[sensitiveColumn]
                    ? `Range: ${ranges[sensitiveColumn].min} - ${ranges[sensitiveColumn].max}`
                    : "Loading range..."}
                </Text>

                {/* Add layers for sensitiveColumn */}
                {layers[sensitiveColumn]?.map((layer, index) => (
                  <Form.Item key={`${sensitiveColumn}-layer-${index}`}>
                    <div style={{ marginBottom: "5px" }}>
                      <label>{`Layer ${index + 1}:`}</label>
                    </div>

                    <Input.Group
                      compact
                      style={{ display: "flex", alignItems: "center" }}
                    >
                      {selectedMethods[sensitiveColumn] === "dates" ? (
                        <RangePicker
                          style={{ width: "85%" }}
                          onChange={(dates, dateStrings) => {
                            updateLayerValue(
                              sensitiveColumn,
                              index,
                              "min",
                              dateStrings[0]
                            );
                            updateLayerValue(
                              sensitiveColumn,
                              index,
                              "max",
                              dateStrings[1]
                            );
                          }}
                        />
                      ) : (
                        <>
                          <Input
                            style={{ width: "40%" }}
                            placeholder={`Min value`}
                            value={layer.min}
                            onChange={(e) =>
                              updateLayerValue(
                                sensitiveColumn,
                                index,
                                "min",
                                e.target.value
                              )
                            }
                          />
                          <span style={{ margin: "0 10px" }}>-</span>
                          <Input
                            style={{ width: "40%" }}
                            placeholder={`Max value`}
                            value={layer.max}
                            onChange={(e) =>
                              updateLayerValue(
                                sensitiveColumn,
                                index,
                                "max",
                                e.target.value
                              )
                            }
                          />
                        </>
                      )}
                      <Button
                        type="primary"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={() => {
                          const updatedLayers = layers[sensitiveColumn].filter(
                            (_, layerIndex) => layerIndex !== index
                          );
                          setLayers({
                            ...layers,
                            [sensitiveColumn]: updatedLayers,
                          });
                        }}
                        style={{ marginLeft: "10px" }}
                      >
                        Delete
                      </Button>
                    </Input.Group>
                  </Form.Item>
                ))}

                <Form.Item>
                  <Button
                    icon={<PlusOutlined />}
                    onClick={() => addLayer(sensitiveColumn)}
                  >
                    Add Layer
                  </Button>
                </Form.Item>
              </>
            )}

          {/* 遮掩字符按钮 */}
          {selectedMethods[sensitiveColumn] === "masking" &&
            maskingData[sensitiveColumn] &&
            maskingData[sensitiveColumn].length > 0 && (
              <div>
                <Text>
                  Making {maskingData[sensitiveColumn].length} characters:
                </Text>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(12, 1fr)", // 每行12个按钮
                    gap: "10px",
                    marginTop: "10px",
                  }}
                >
                  {maskingData[sensitiveColumn].map((char, index) => (
                    <Button
                      key={index}
                      type={char === "*" ? "primary" : "default"}
                      onClick={() =>
                        handleToggleMasking(sensitiveColumn, index)
                      }
                    >
                      {char}
                    </Button>
                  ))}
                </div>
              </div>
            )}

          {/* <Form.Item>
            <Checkbox>I accept the terms</Checkbox>
            <Link href="#" target="_blank">
              Read our T&Cs
            </Link>
          </Form.Item> */}

          <Form.Item>
            <Button
              htmlType="submit"
              style={{
                width: "100%",
                color: "white",
                backgroundColor: "black",
                transition: "background-color 0.3s, color 0.3s",
              }}
              block
            >
              Submit Hierarchy Info
            </Button>
          </Form.Item>
        </Form>
      </Content>
    </Layout>
  );
};

export default HierarchyForm;
