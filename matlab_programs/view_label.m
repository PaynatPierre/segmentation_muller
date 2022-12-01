%M = SHGZ;
%M = TPEFZ;
M = Deat_diffZ;



Mglob = zeros(500,500);

[ha, pos] = tight_subplot(1,1,[.001 .001],[.001 .001],[.001 .001]); 
ii = 0
for i = 1:1
    for j = 1:1
        Mij = M(:,:,i,j);
        ii = ii+1;
        axes(ha(ii));
        imagesc(Mij);
        %set(ha(1:4),'XTickLabel',''); set(ha,'YTickLabel','')
        %figure()
        %imagesc(Mij)
        %title( strcat(num2str(i),num2str(j) ) )
        %Mglob((i-1)*500 +1 : i*500, (j-1)*500 +1 : j*500) = Mij;
        
    end
end
set(ha(1:16),'XTickLabel',''); set(ha,'YTickLabel','')
