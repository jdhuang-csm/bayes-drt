function visualize(data,Zhat)

subplot(1,2,1)
plot(real(Zhat),-imag(Zhat),data(:,2),-data(:,3))

subplot(2,2,2)
plot(log10(data(:,1)),real(Zhat),log10(data(:,1)),data(:,2));

subplot(2,2,4)
plot(log10(data(:,1)),-imag(Zhat),log10(data(:,1)),-data(:,3));

end