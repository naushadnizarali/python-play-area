import openai
import os

# Your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# BRD document
brd_document = """KEYWORD: software
Software Requirements Specification Amazing Lunch Indicator Sarah Geagea Sheng Zhang Niclas Sahlin Faegheh Hasibi Farhan Hameed Elmira Rafiyan Magnus Ekberg . + 
The purpose of this document is to give a detailed description of the requirements for the Amazing Lunch Indicator ALI software. + 
Furthermore, the software needs both Internet and GPS connection to fetch and display results. + 
The software also interacts with the GPSNavigator software which is required to be an already installed application on the users mobile phone. + 
The software also interacts with the GPSNavigator software which is required to be an already installed application on the users mobile phone. + 
Software interfaces The mobile application communicates with the GPS application in order to get geographical information .. + 
Design constraints This section includes the design constraints on the software caused by the hardware. + 
Software system attributes The requirements in this section specify the required reliability, availability, security and maintainability of the software system. + 
Software system attributes The requirements in this section specify the required reliability, availability, security and maintainability of the software system. + 


KEYWORD: scope
This section gives a scope description and overview of everything included in this SRS document. + 
Scope + 


KEYWORD: constraints
It will also explain system constraints, interface and interactions with other external applications. + 
At last, the constraints and assumptions for the system will be presented. . + 
Constraints The mobile application is constrained by the system interface to the GPS navigation system within the mobile phone. + 
Design constraints This section includes the design constraints on the software caused by the hardware. + 
Design constraints This section includes the design constraints on the software caused by the hardware. + 


KEYWORD: database
All system information is maintained in a database, which is located on a webserver. + 
For that, a database will be used. + 
Both the mobile application and web portal will communicate with the database, however in slightly different ways. + 
The mobile application will only use the database to get data while the web portal will also add and modify data. + 
All of the database communication will go over the Internet. + 
Since the application fetches data from the database over the Internet, it is crucial that there is an Internet connection for the application to function. + 
Both the web portal and the mobile application will be constrained by the capacity of the database. + 
Since the database is shared between both application it may be forced to queue incoming requests and therefor increase the time it takes to fetch data. . + 
The physical GPS is managed by the GPS application in the mobile phone and the hardware connection to the database server is managed by the underlying operating system on the mobile phone and the web server. .. + 


KEYWORD: stakeholders
It will also describe what type of stakeholders that will use the system and what functionality is available for each type. + 


KEYWORD: assumptions
At last, the constraints and assumptions for the system will be presented. . + 
Assumptions and dependencies One assumption about the product is that it will always be used on mobile phones that have enough performance. + 


KEYWORD: dependencies
Assumptions and dependencies One assumption about the product is that it will always be used on mobile phones that have enough performance. + 


KEYWORD: performance
Assumptions and dependencies One assumption about the product is that it will always be used on mobile phones that have enough performance. + 
METER Observations done from the performance log during testing MUST No more than MB. + 


KEYWORD: hardware
If the phone does not have enough hardware resources available for the application, for example the users might have allocated them with other applications, there may be scenarios where the application does not work as intended or even at all. + 
Hardware interfaces Since neither the mobile application nor the web portal have any designated hardware, it does not have any direct hardware interfaces. + 
Hardware interfaces Since neither the mobile application nor the web portal have any designated hardware, it does not have any direct hardware interfaces. + 
Hardware interfaces Since neither the mobile application nor the web portal have any designated hardware, it does not have any direct hardware interfaces. + 
The physical GPS is managed by the GPS application in the mobile phone and the hardware connection to the database server is managed by the underlying operating system on the mobile phone and the web server. .. + 
Design constraints This section includes the design constraints on the software caused by the hardware. + 


KEYWORD: testing
METER Measurements obtained from hours of usage during testing. + 
METER Observations done from the performance log during testing MUST No more than MB. + 
DEP none ID QR21 TITLE Application testability DESC Test environments should be built for the application to allow testing of the applications different functions. + 


KEYWORD: reliability
Software system attributes The requirements in this section specify the required reliability, availability, security and maintainability of the software system. + 
Reliability ID QR9 TAG + 
This will be considered and reliability continuously improved during the whole process. + 


KEYWORD: availability
Software system attributes The requirements in this section specify the required reliability, availability, security and maintainability of the software system. + 
This will be considered and availability continuously improved during the whole process. + 


KEYWORD: security
Software system attributes The requirements in this section specify the required reliability, availability, security and maintainability of the software system. + 
QR12 Communicatio As the system grows, and with n security respect for the users this need to be included. + 
QR13 FR22 Nonexisting This need to be included in the first account release to enhance the safety of the security for system. + 
Owners QR14 Nonexisting This needs to be included in the first account release to enhance the safety of the security for the system. + 
administrators QR15 FR23 Log in security This needs to be included in the first for rest. + 
QR16 Log in security This need to be included in the first for release to prevent illegitimate administrators attempts to use the administrator account. + 
QR17 FR3 User account This needs to be included in the first creation release to resolve conflict between security users with the same name. + 
This needs to be included in the first owner account release to resolve conflict between creation restaurant owners with the same security name. + 


KEYWORD: maintainability
Software system attributes The requirements in this section specify the required reliability, availability, security and maintainability of the software system. + 
SystemReliability .. Maintainability ID QR19 TITLE Application extendibility DESC The application should be easy to extend. + 


KEYWORD: portability
Portability ID QR20 TITLE Application portability DESC The application should be portable with iOS and Android. + 
Portability ID QR20 TITLE Application portability DESC The application should be portable with iOS and Android. + 
This will be considered and portability continuously improved during the whole process. + 


KEYWORD: budget
This is not in some way vital for the application and can be discarded if the project gets delayed or overruns the budget. + 
"""


# Function to generate estimates using GPT-3
def generate_estimates(brd_document):
    # Prompt to provide context for the LM
    # prompt = f"Generate estimates for the cost and timeline based on the following BRD document:\n{brd_document}"
    prompt = f"Can you please list down number of ui screens and their complexities, including number of backend services and their complexities, plus any external integrations that is defined in the context.\n CONTEXT:\n{brd_document} \nPlease format the result in a presentable format."

    # Generate response using GPT-3
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant to a product owner.",
            },
            {"role": "user", "content": prompt},
        ],
        # max_tokens=150,
        # n=1,
        # stop=None,
        temperature=0.5,
    )

    return response.choices[0].message.content


# Generate estimates using GPT-3
estimation_result = generate_estimates(brd_document)

# Print the generated estimates
print("Estimated cost and timeline based on BRD document:\n", estimation_result)
