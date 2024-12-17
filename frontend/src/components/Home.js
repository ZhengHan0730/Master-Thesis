import React from "react";
import AppHeader from "./Header"; // Import the Header component
import { Layout, Typography, Card, Row, Col } from "antd";
import "../index.css"; // Make sure to include CSS for flipping cards

const { Title,Paragraph } = Typography;
const { Meta } = Card;

const Home = () => {
  const cardsData = [
    {
      title: "k-anonymity",
      image: "k.png",
      description: "k-Anonymity is a data anonymization technique that ensures each record in a dataset is indistinguishable from at least \( k-1 \) other records based on quasi-identifiers. This helps minimize the risk of re-identifying individuals, thereby protecting their privacy."
    },
    {
      title: "l-diversity",
      image: "l.png",
      description: "l-Diversity is an enhancement of k-anonymity aimed at addressing its limitations. It ensures that within each group of indistinguishable records, there is sufficient diversity in sensitive attributes. This helps prevent attribute disclosure by making it harder to infer sensitive information about individuals."
    },
    {
      title: "t-closeness",
      image: "t.png",
      description: "t-Closeness is a data anonymization model that further improves upon l-diversity. It ensures that the distribution of a sensitive attribute within each anonymized group is close to its distribution in the overall dataset. This prevents attackers from deducing sensitive information by comparing group distributions with the global distribution."
    },
    {
      title: "km-anonymity",
      image: "m.png",
      description: "km-Anonymity combines k-anonymity with masking techniques to provide more robust data protection. It generalizes data to achieve k-anonymity while also applying masking to specific values to prevent attribute and identity disclosure, enhancing the privacy of more sensitive datasets."
    }
  ];

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <AppHeader />
      <div style={{ textAlign: 'center', marginBottom: '40px', marginTop: '100px' }}>
        <Title level={1} style={{ fontSize: '500%' }}>Welcome to PREDANO</Title>
        <Paragraph style={{ maxWidth: '800px', margin: '20px auto', fontSize:'1.3rem' }}>
        Our data anonymization platform is developed in a Python environment, offering high scalability. For academic researchers, Python is more familiar and easier to maintain compared to other backend languages like Java. Additionally, Python facilitates deeper machine learning analyses. The platform also supports easy integration of new algorithms. For users, we provide a convenient and user-friendly web-based data anonymization application. And the following algorithms are currently intergrated in our platform:
        </Paragraph>
      </div>
      <Row gutter={[16, 16]} justify="center" style={{ padding: '20px' }}>
        {cardsData.map((card, index) => (
            <Col xs={24} sm={12} md={12} lg={6} key={index}>
  <div className="flip-card" style={{ height: '350px', width: '100%' }}>
    <div className="flip-card-inner">
      <div className="flip-card-front">
        <Card
          cover={<img alt={card.title} src={card.image} style={{ height: '250px', objectFit: 'cover' }} />}
          bordered={false}
          style={{ height: '100%', width: '100%' }}
        >
          <Meta  title={card.title} />
        </Card>
      </div>
      <div className="flip-card-back" style={{ height: '350px', width: '100%' }}>
        <div style={{ padding: '15px', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%',fontSize:'1.2rem' }}>
          <p>{card.description}</p>
        </div>
      </div>
    </div>
  </div>
</Col>

        ))}
      </Row>
    </Layout>
  );
};

export default Home;