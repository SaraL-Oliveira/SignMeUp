#include <WebServer.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Arduino.h>

//#include "AsyncUDP.h" // To Manage IP broadcast

// For the MPU
#include <Wire.h>
#include <I2Cdev.h>
#include <MPU6050.h>

//IPAddress broadcastIp(192, 168, 1, 255);
IPAddress broadcastIp(239,255,255,250);

//set up to connect to an existing network (e.g. mobile hotspot from laptop that will run the python code)
const char* ssid = "MEO-282930"; //"Vodafone-B30051"; // "MEO-FIBRA";
const char* password = "ce7a191c6d"; //"9D897A7947"; // "62683acff5";
WiFiUDP Udp;
unsigned int localUdpPort = 4211;  //  port to listen on - USE 4210 for RIGHT HAND and 4211 for LEFT HAND
String HAND = "Left";
char incomingPacket[255];  // buffer for incoming packets (messages), size we can define bigger if needed

// For the MPU
MPU6050 mpu; //Module declaration.
int16_t ax, ay, az, gx, gy, gz; //Variable declaration. int16_t is a 16 bit  //int variable without signal. 
//Variables with 'a' are accelerometer's axis and 'g' are gyroscope's axis.

//sample frequency and period (Hz and s)
int freq_s = 50, ps = 0;
unsigned long currentTime = 0, previousTime = 0;


void sendMsg(char msg[255], WiFiUDP *Udp) {
  // once we know where we got the inital packet from, send data back to that IP address and port
  (*Udp).beginPacket((*Udp).remoteIP(), (*Udp).remotePort());
  (*Udp).printf(msg);
  (*Udp).endPacket();
}

void broadcast(WiFiUDP *Udp, IPAddress broadcastIp, char my_ip[255]) {
  (*Udp).beginPacket(broadcastIp, localUdpPort);
  (*Udp).printf(my_ip);
  (*Udp).endPacket();
}

void setup()
{
  int status = WL_IDLE_STATUS;
  //Serial.begin(115200);

  Wire.begin();  //Initialize I2C communication which is used between SeedUino and MPU
  mpu.initialize(); //Initialize the module.

  //sample period in ms
  ps = int(1000 * 1/ (float) freq_s);

  WiFi.begin(ssid, password);
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    //Serial.print(".");
  }
  //Serial.println("Connected to wifi");
  Udp.begin(localUdpPort);
  //Serial.printf("Now listening at IP %s, UDP port %d\n", WiFi.localIP().toString().c_str(), localUdpPort);

  bool readPacket = false;
  while (!readPacket) {
    int packetSize = Udp.parsePacket();
    if (packetSize)
     {
      // receive incoming UDP packets
      //Serial.printf("Received %d bytes from %s, port %d\n", packetSize, Udp.remoteIP().toString().c_str(), Udp.remotePort());
      int len = Udp.read(incomingPacket, 255);
      if (len > 0) { incomingPacket[len] = 0; }
      //Serial.printf("UDP packet contents: %s\n", incomingPacket);
      readPacket = true;
    } 
  }
  delay(4000);
  sendMsg((char*)(WiFi.localIP().toString().c_str()), &Udp);
}


void loop()
{
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz); //Function that makes the readings of accelerometer and gyroscope axis. 
  
  // Sends data at a period, ps, at a time
  currentTime = millis();
  if(currentTime - previousTime >= ps){
    // Just sends the data
    char dataMsg[255]; //= string(currentTime) + "," + string(ax) + "," + string(ay) + "," + string(az) + "," + string(gx) + "," + string(gy) + "," + string(gz)
    sprintf(dataMsg, "%s,%d,%d,%d,%d,%d,%d,%d", HAND, currentTime, ax, ay, az, gx, gy, gz);
    //Serial.print(HAND);Serial.print(",");Serial.print(currentTime); Serial.print(","); Serial.print(ax);  Serial.print(",");  Serial.print(ay);  Serial.print(",");  Serial.print(az);  Serial.print(",");  Serial.print(gx);  Serial.print(",");  Serial.print(gy);  Serial.print(","); Serial.println(gz);
    sendMsg(dataMsg, &Udp);
    previousTime = currentTime;
  }
  //delay(1000);
}