import React from 'react';
import AppHeader from "./Header"; // Import the Header component

import { Layout, Card, Typography } from "antd";

const { Title, Paragraph } = Typography;
const TermsAndConditions = () => {
  return (<Layout style={{ minHeight: "100vh" }}>
    <AppHeader/>
    <div style={{ padding: '20px' }}>
      <h1>Terms and Conditions</h1>
        <Paragraph style={{ textAlign: 'left', maxWidth: '800px' }}>
          <strong>1. Introduction</strong><br />
          Welcome to PREDANO ("Platform"). By accessing or using our Platform, you agree to comply with and be bound by these Terms and Conditions ("Terms"). If you do not agree with these Terms, please do not use the Platform.
        </Paragraph>
        <Paragraph style={{ textAlign: 'left', maxWidth: '800px' }}>
          <strong>2. Definitions</strong><br />
          "User": Any individual or entity accessing or using the Platform.<br />
          "Personal Data": Any information relating to an identified or identifiable natural person.<br />
          "Anonymized Data": Data that has been processed to remove personally identifiable information, rendering it unidentifiable.
        </Paragraph>
        <Paragraph style={{ textAlign: 'left', maxWidth: '800px' }}>
          <strong>3. User Obligations</strong><br />
          <strong>Compliance</strong>: Users must comply with all applicable laws and regulations, including data protection laws such as GDPR.<br />
          <strong>Data Ownership</strong>: Users retain ownership of their data and are responsible for its legality and appropriateness.<br />
          <strong>Prohibited Activities</strong>: Users shall not upload malicious code, engage in unlawful activities, or misuse the Platform in any manner.
        </Paragraph>
        <Paragraph style={{ textAlign: 'left', maxWidth: '800px' }}>
          <strong>4. Platform Usage</strong><br />
          <strong>License</strong>: GPL-2.<br />
          <strong>Modifications</strong>: We reserve the right to modify or discontinue the Platform or its features at any time without prior notice.
        </Paragraph>
        <Paragraph style={{ textAlign: 'left', maxWidth: '800px' }}>
          <strong>5. Changes to Terms</strong><br />
We reserve the right to update these Terms at any time. 
        </Paragraph>
        <Paragraph style={{ textAlign: 'left', maxWidth: '800px' }}>
          <strong>6. Contact Information</strong><br />For questions or concerns regarding these Terms, please contact us at 123@uzh.ch.
        </Paragraph>
    </div></Layout>
  );
};

export default TermsAndConditions;
