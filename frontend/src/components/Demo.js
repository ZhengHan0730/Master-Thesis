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
              <Paragraph><strong>Select dataset splite and set the splite ratio</strong><br />
              Enable to split the original dataset into training and test sets before anonymization. </Paragraph>
            </li>
            <li>
              <Paragraph><strong>Set Values</strong><br />
              Enter values as needed for your chosen algorithm.</Paragraph>
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
            <img src="algorithm.png" alt="Application Page Screenshot" style={{ maxWidth: '30%', height: 'auto' }} />
          </div>
          </ol>

          <Title level={3}>Viewing Results</Title>
          <Paragraph>
            After submission, you will be able to review the anonymized dataset. You can download the processed data for further analysis or integration into your projects.
          </Paragraph>
          <Paragraph style={{ textAlign: 'left', maxWidth: '800px', margin: '0 auto' }}>
          

          <Title level={3}>Evaluation of Anonymized Data</Title>
          <ol>
            <li>
              <Paragraph><strong>Select Evaluation Methods</strong><br />
              Choose statistical and machine learning methods to evaluate the quality of your anonymized dataset. Options include mean, median, variance, Wasserstein distance, and more.</Paragraph>
              <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <img src="select_methods.png" alt="Select Evaluation Methods" style={{ maxWidth: '100%', height: 'auto' }} />
              </div>
            </li>

            <li>
              <Paragraph><strong>Upload Your Data Files</strong><br />
              Upload both your original dataset and the anonymized dataset for comparison. Files must be in CSV or TSV format.</Paragraph>
              <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <img src="uploads_files.png" alt="Upload Data Files" style={{ maxWidth: '100%', height: 'auto' }} />
              </div>
            </li>

            <li>
              <Paragraph><strong>Select Columns for Evaluation</strong><br />
              Specify the columns that you want to evaluate. Typically, these include quasi-identifiers or sensitive attributes.</Paragraph>
              <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <img src="select_columns.png" alt="Select Columns for Evaluation" style={{ maxWidth: '100%', height: 'auto' }} />
              </div>
            </li>

            <li>
              <Paragraph><strong>Review and Submit</strong><br />
              Review the selected methods, uploaded files, and columns before starting the evaluation. Confirm your selections and click "Submit Evaluation" to begin.</Paragraph>
              <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <img src="review_and_submit.png" alt="Review and Submit" style={{ maxWidth: '100%', height: 'auto' }} />
              </div>
            </li>

            <li>
              <Paragraph><strong>View Evaluation Results</strong><br />
              After processing, the results page displays metrics comparing original and anonymized data, showing differences in statistical values and errors. You can download these results for further examination.</Paragraph>
              <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <img src="evaluation_result.png" alt="Evaluation Results" style={{ maxWidth: '100%', height: 'auto' }} />
              </div>
            </li>
          </ol>
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

