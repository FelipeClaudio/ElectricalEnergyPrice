figure
startDate = datenum(2003, 05, 31);
endDate = datenum(2018, 09, 30);
xData = linspace(startDate,endDate,size(SE, 1));
ax = gca;
ax.XTick = xData;
plot(xData, SE, '-bx')
legend('PLD Medio')
title('PLD Medio SE - 05/2003 - 09/2018')
datetick('x','dd/mm/yy','keepticks')
