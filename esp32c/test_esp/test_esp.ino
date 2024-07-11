#include <WebServer.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Arduino.h>

#include "AsyncUDP.h" // To Manage IP broadcast

// For the MPU
#include <Wire.h>
#include <I2Cdev.h>
#include <MPU6050.h>

//IPAddress broadcastIp(192, 168, 1, 255);
IPAddress broadcastIp(239,255,255,250);

//set up to connect to an existing network (e.g. mobile hotspot from laptop that will run the python code)
const char* ssid = "MEO-FIBRA"; //"Vodafone-B30051";
const char* password = "62683acff5"; //"9D897A7947";
AsyncUDP  Udp;
const char* udpMsg;
unsigned int localUdpPort = 4210;  //  port to listen on
//har incomingPacket[255];  // buffer for incoming packets (messages), size we can define bigger if needed

// For the MPU
MPU6050 mpu; //Module declaration.
int16_t ax, ay, az, gx, gy, gz; //Variable declaration. int16_t is a 16 bit  //int variable without signal. 
//Variables with 'a' are accelerometer's axis and 'g' are gyroscope's axis.

//sample frequency and period (Hz and s)
int freq_s = 1, ps = 0;
unsigned long currentTime = 0, previousTime = 0;

/*
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
*/

void setup()
{
  int status = WL_IDLE_STATUS;
  Serial.begin(115200);

  Wire.begin();  //Initialize I2C communication which is used between SeedUino and MPU
  mpu.initialize(); //Initialize the module.
  /* if(!mpu.testConnection()){
    Serial.print("Connected");
    while (1); 
    } */

  //sample period in ms
  ps = int(1000 * 1/ (float) freq_s);

  WiFi.begin(ssid, password);
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to wifi");
  
  Udp.connect(IPAddress(192,168,1,255), 4210);

  //Udp.begin(localUdpPort);
  Serial.printf("Now listening at IP %s, UDP port %d\n", WiFi.localIP().toString().c_str(), localUdpPort);

  //Next part is just to setup the wifi connection, we receive one msg from PC and then send one msg to PC
  // If both work then we're good to go
  // we recv one packet from the remote so we can know its IP and port
  int count = 1000;
  while(count > 0) {
    Udp.broadcastTo(WiFi.localIP().toString().c_str(), 4210);
    Serial.print("Sending Broadcast: ");
    Serial.println(count);
    delay(2000);
    count = count - 1;
  }

  
}


void loop()
{
  Serial.println("Doing...");
  delay(2000);
}

// Previously to send msg inside loop()
/* Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
  Udp.printf("Hello SeedUino");
  Udp.endPacket(); */