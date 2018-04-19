function angle = clcAngle(difference)
if difference(1) == 0
    if difference(2) == 0
        angle = 0;
    else
        if difference(2) > 0
            angle = pi/2;
        else
            angle = -pi/2;
        end
    end
    return
end
angle = atan2(difference(2),difference(1));


% angle = atand(difference(2)/difference(1))/180*pi;
% if(difference(1)<0)
%    angle = angle;
% else
%    if(difference(2)<0)
%        angle = angle - 2* pi;
%    end
% end