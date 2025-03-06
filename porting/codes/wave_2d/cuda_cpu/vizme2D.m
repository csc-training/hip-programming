clear
% Load the DATA and infos
nxy = load('infos.inf');  PRECIS=nxy(1); nx=nxy(2); ny=nxy(3);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end 
id = fopen('0_P.res' ); P  = fread(id,DAT); fclose(id); P  = reshape(P ,nx  ,ny  );
id = fopen('0_Vx.res'); Vx = fread(id,DAT); fclose(id); Vx = reshape(Vx,nx+1,ny  );
id = fopen('0_Vy.res'); Vy = fread(id,DAT); fclose(id); Vy = reshape(Vy,nx  ,ny+1);
% Plot it
figure(2),clf,
subplot(311),imagesc(flipud(P' )),axis image,colorbar
subplot(312),imagesc(flipud(Vx')),axis image,colorbar
subplot(313),imagesc(flipud(Vy')),axis image,colorbar
