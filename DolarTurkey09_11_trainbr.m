clear all;
clc;

%% Load Data
d=load('DolarTurkey09_11.mat');
d=d.DolarTurkey09_11; %katmanlı olduğu için d'de olan datayı yine d'ye attım
%d=diff(log(d));
%% Prepearing data
%algoritma verilerini 4 katmalı inputla bir tane de target ile
%hazırladım. Algoritmaya veriyi tek sütunlu matris şeklinde veremeyiz. Bu
%yüzden target ve inputları hazırlamamız lazım.
target=d(53:end);
x52=d(1:end-52);
x26=d(27:end-26);
x17=d(36:end-17);
x1=d(52:end-1);
input=[x52 x26 x17 x1];
data=[input target]; %inputlarımla targetımı birleştirdim ve veriyi hazırlamış oldum
ndata=size(data,1); %lenght(data) yerine ndata tanımladım.

%% Neural Network
%ysa'nın bilgiyi öğrenip öğrenmediğini anlamak için ysa yı bu veri ile
%sınava sokmamız gerekiyor. Bizim elimizde bir tane veri var. Bu veriyi
%sınav için bölebiliriz.
%bizim 500 satırlı verimiz var .Bu verinin 47 tanesini sınav için ayıralım.
%geri kalanı ile de ysa ya öğretelim.

traindata=data(1:ndata-47,:); %bu kadar train datası olsun
traininputs=(traindata(:,1:end-1))'; %train inputları ayırmak için sondakı target sütununu çıkardım 
traintarget=(traindata(:,end))'; %son sütun traintarget

testdata=data(end-46:end,:);  %test data 47 tane satırdan oluşuyor
testinputs=(testdata(:,1:end-1))'; %test inputları ayırdım.
testtarget=(testdata(:,end))'; %son sütun test target
%network lar girdileri sütun olarak alırlar. O yüzden transpose larını aldım. 
%% Creating Network
%burda ysa'nın yapısını oluşturuyoruz
layers=[]; %katman sayısını değiştirebiliriz
transferfun={'logsig'};%transfer fonksiyonu
trainFcn = 'trainbr';%training fonksiyonu

%newfit komutu input ve target vektörü alır. 
ag=newfit(traininputs,traintarget,layers,transferfun,trainFcn);%newfit komutu ile ağ oluşturuyoruz.
%train komutuyla ag network'ünü eğitiyoruz
%enetwork burda yeni eğitim almış ağ oluyor.
enetwork=train(ag,traininputs,traintarget);

%sınavdan geçirmek için sim komutunu kullanıyoruz. Yani veriyi bir
%simülasyona sokuyoruz.
sinavtr=sim(enetwork,traininputs);
hatatr=(sinavtr-traintarget)/100; %ysa'nın yaptığı hata yüzdeliği.
%hatatr=gördüğü data için hata yüzdesi. Tabiki de bunu hata yüzdesi daha az
%olacak
sinavts=sim(enetwork,testinputs);
hatats=(sinavts-testtarget)/100; %ysa'nın yaptığı hata yüzdeliği
%hatats=görmediği data için hata yüzdesi
%% Grafikler  train data
%grafikler ile detaylandırdım.Bu grafikler sayesinde Neural Network'ün ne
%kadar iyi çalıştığını gösterdim.
figure;
 subplot(2,2,1)
 plot(traintarget,'b')
 title("Neural Network'ün gördüğü veri");
 hold on;
 plot(sinavtr,'r');
 legend('gerçek veri',"Neural Network'den gelen sonuç");
  subplot(2,2,3);
 plot(hatatr);
 legend('hatalar');

%% Grafik test data


subplot(2,2,2)
plot(testtarget,'b')
title("Neural Network'ün görmediği veri");
hold on;
plot(sinavts,'r');
legend('gerçek veri',"Neural Network'den gelen sonuç");

subplot(2,2,4);
plot(hatats);
legend('hatalar');

%%
%hold on;
%regresyon grafiği oluşturdum
figure;
plotregression(testtarget,enetwork(testinputs),'Regression');
%figure;
%hold on;
%plotregression(traintarget,enetwork(traininputs),'Regression');