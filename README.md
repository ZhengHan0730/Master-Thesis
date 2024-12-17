# Developer Guideline

This document provides the guidelines for users to launch the application from both frontend and backend which including two parts, one is the installation and run the Data Anonymity Tool (PREDANO) in Apple's MacOS operation system. For other operating systems, such as UNIX or Windows, you can search online for operations similar to the macOS procedures described below. My data anonymization platform (PREDANO) can be obtained via the GitHub address.Alternatively, the platform is also accessible through the GitLab address in UZH IFI.

## 1. setting up for Frontend 

Because our front-end is built using the React framework, we need a JavaScript runtime environment called Node. 

Environmental preparation: 
Install Node.js and npm: This is the runtime environment for React projects, and you can download the installation package from the Node.js official website. ‌ 

```
https://nodejs.org/
```

Select the code editor: 

We recommend using Visual Studio Code, which provides rich extensions and powerful features. ‌

```
https://code.visualstudio.com
```

Get source code:

```
cd dir
git clone url
```

Download the library dependencies of node_modules: 

Because our code depends on other libraries, we need to install the environment in advance, so we need to perform the following operations. 

```
npm install
```

Enter the project directory and run the npm start command to start the local development server and view the React project, which is our PREDANO frontend page.

```
npm start
```

By following the above steps, users can successfully run our PREDANO React frontend project.

## 2. setting up for Backend

Users not only need to run the existing frontend, but also need to run the backend interface. To run our pre written Flask PREDANO application from 0 to 1, the following steps need to be followed:

Configure Environment:
Ensure that Python and pip package management tools are installed.
Install Flask framework using pip: 

```
pip install Flask
```

Install depencies packages:

Download some environment packages required to run our backend Flask program from the requirements. txt document

```
pip install −r requirements . txt
```

Running Flask application：
Open the command line and enter the project directory. Execute the  command to start the Flask application.

```
flask run --without-threads
```

Accessing the application:
Open a web browser and enter url Access Flask application. Confirm that the expected response content is displayed in the browser.

```
http://127.0.0.1:3000/
http://localhost:3000/
```

Follow the above steps to successfully run our written Data Anonymity Tool (PREDANO) backend Flask application from 0 to 1.
