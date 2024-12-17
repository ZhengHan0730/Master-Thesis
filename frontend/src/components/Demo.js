import React from 'react';
import AppHeader from "./Header"; // Import the Header component
import { Layout, Card, Typography } from "antd";

const { Title, Paragraph,Link } = Typography;

const Demo = () => {
  return (
    <Layout style={{ minHeight: "100vh", backgroundColor: "#f0f2f5" }}>
      <AppHeader />
      <div style={{ padding: '20px' }}>
        <Card style={{ maxWidth: '800px', margin: 'auto' }}>
          <Title style={{textAlign: 'center'}} level={1}>Demo</Title>

          <Title level={3}>User Instruction</Title>
          <ol>
            <li>
              <Paragraph><strong>Navigate to the Application Page</strong><br />
              Access the platform and go to the main application page where you will find the data processing interface.</Paragraph>
            </li>
                <div style={{ textAlign: 'center', margin: '20px 0' }}>
            <img src="application.png" alt="Application Page Screenshot" style={{ maxWidth: '100%', height: 'auto' }} />
          </div>
          <li><strong>Upload Your Dataset</strong><br />Click the button labeled "Click to Upload CSV/TSV" to upload your dataset in CSV or TSV format. Ensure your dataset is correctly formatted for processing.</li>

            <li>
              <Paragraph><strong>Select Identifiers</strong><br />
              Choose the columns in your dataset that represent identifiers. These may include unique information that can directly identify an individual.</Paragraph>
            </li>
            <li>
              <Paragraph><strong>Select Quasi-Identifiers</strong><br />
              Select the columns that could potentially identify an individual when combined with other information.</Paragraph>
            </li>
            <li>
              <Paragraph><strong>Select Sensitive Attributes</strong><br />
              Identify and select the columns that contain sensitive information requiring protection.</Paragraph>
            </li>
            <li>
              <Paragraph><strong>Accept the Terms</strong><br />
              Confirm that you accept the terms and conditions before proceeding.</Paragraph>
            </li>
            <li>
              <Paragraph><strong>Submit for Processing</strong><br />
              Click the "Submit Processing Info" button to start the anonymization process. The platform will process your dataset based on your selected parameters.</Paragraph>
            </li>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
            <img src="processing.png" alt="Application Page Screenshot" style={{ maxWidth: '60%', height: 'auto' }} />
          </div>
          </ol>

          <Title level={3}>Applying Hierarchies for Quasi-Identifiers</Title>
          <Paragraph><strong>Customize the Anonymization Methods:</strong><br />
          Choose the anonymization methods for each quasi-identifier. Options may include categorization, ordering, or date handling. Add layers as needed to fine-tune the granularity of data generalization.</Paragraph>
          <div style={{ textAlign: 'center', margin: '20px 0' }}>
            <img src="hierarchy.png" alt="Application Page Screenshot" style={{ maxWidth: '60%', height: 'auto' }} />
          </div>
          <Title level={3}>Configure Anonymization Algorithm</Title>
          <ol>
            <li>
              <Paragraph><strong>Select the Algorithm</strong><br />
              Choose from supported algorithms such as k-Anonymity, l-Diversity, or t-Closeness.</Paragraph>
            </li>
            <li>
              <Paragraph><strong>Set Values</strong><br />
              Enter values for k (k-value) or l (l-value) as needed for your chosen algorithm.</Paragraph>
            </li>
            <li>
              <Paragraph><strong>Adjust Suppression Rate</strong><br />
              Set the suppression rate to control the amount of data removed to meet anonymization requirements.</Paragraph>
            </li>
            <li>
              <Paragraph><strong>Submit</strong><br />
              Click the "Submit" button to apply the selected algorithm and view the processed results.</Paragraph>
            </li>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
            <img src="algorithm.png" alt="Application Page Screenshot" style={{ maxWidth: '100%', height: 'auto' }} />
          </div>
          </ol>

          <Title level={3}>Viewing Results</Title>
          <Paragraph>
            After submission, you will be able to review the anonymized dataset. You can download the processed data for further analysis or integration into your projects.
          </Paragraph>
          <Paragraph style={{ textAlign: 'left', maxWidth: '800px', margin: '0 auto' }}>
          <Link href="Developer Guidelines.pdf" download>
            Developer guideline help file download.
          </Link>
        </Paragraph>
        </Card>
      </div>
    </Layout>
  );
};

export default Demo;

