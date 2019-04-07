# convex relaxation (envelope) of cos function
function cv_cos(x,xL,xU)
  r = 0.0
  kL = Base.ceil((-0.5-xL/(2.0*pi)))
  if (x<=(pi-2.0*pi*kL))
    xL1 = xL+2.0*pi*kL
    if (xL1 >= pi/2.0)
      return cos(x),-sin(x)
    end
    xU1 = min(xU+2.0*pi*kL,pi)
    if ((xL1>=(-pi/2))&&(xU1<=(pi/2)))
      if (abs(xL-xU)<MC_param.env_tol)
        r = 0.0
      else
        r = (cos(xU)-cos(xL))/(xU-xL)
      end
      return cos(xL)+r*(x-xL),r
    end
    return cv_cosin(x+(2.0*pi)*kL,xL1,xU1)
  end
  kU = Base.floor((0.5-xU/(2.0*pi)))
  if (x>=(-pi-2.0*pi*kU))
    xU2 = xU+2.0*pi*kU
    if (xU2<=-pi/2.0)
      return cos(x),-sin(x)
    end
    return cv_cosin(x+2.0*pi*kU,max(xL+2.0*pi*kU,-pi),xU2)
  end
  return -1.0,0.0
end
# function for computing convex relaxation over nonconvex and nonconcave regions
function cv_cosin(x,xL,xU)
  xj = -Inf
  if (abs(xL)<=abs(xU))
    left = false
    x0 = xU
    xm = xL
  else
    left = true
    x0 = xL
    xm = xU
  end
  try
    xj = newton(x0,xL,xU,cv_cosenv,dcv_cosenv,xm,0.0)
  catch e
    if isa(e, ErrorException)
      xj = golden_section(xL,xU,cv_cosenv,xm,0.0)
    end
  end
  if ((left && x<=xj)||((~left) && x>=xj))
    return cos(x),-sin(x)
  else
    if abs(xm-xj)<MC_param.env_tol
      r = 0.0
    else
      r = (cos(xm)-cos(xj))/(xm-xj)
    end
    return cos(xm)+r*(x-xm),r
  end
end

# pivot point calculation function for convex relaxation of cosine
cv_cosenv(x,y,z) = (x-y)*sin(x)+cos(x)-cos(y)
dcv_cosenv(x,y,z) = (x-y)*cos(x)

# concave relaxation (envelope) of cos function
function cc_cos(x,xL,xU)
  temp = cv_cos(x-pi,xL-pi,xU-pi)
  return -temp[1],-temp[2]
end
function cos_arg(xL,xU)
  kL = Base.ceil(-0.5-xL/(2.0*pi))
  xL1 = xL+2.0*pi*kL
  xU1 = xU+2.0*pi*kL
  if ~((xL1>=-pi)&&(xL1<=pi))
    error("Cosine Argument Calculation: xL out of bounds.")
  end
  if (xL1 <= 0.0)
    if (xU1 <= 0.0)
      arg1 = xL
      arg2 = xU
    elseif (xU1 >= pi)
      arg1 = pi-2.0*pi*kL
      arg2 = -2.0*pi*kL
    else
      arg1 = (cos(xL1) <= cos(xU1)) ? xL : xU
      arg2 = -2.0*pi*kL
    end
  end
  if (xU1 <= pi)
    arg1 = xU
    arg2 = xL
  elseif (xU1 >= (2.0*pi))
    arg1 = pi-2.0*pi*kL
    arg2 = 2.0*pi-2.0*pi*kL
  else
    arg1 = pi-2.0*pi*kL
    arg2 = (cos(xL1) >= cos(xU1)) ? xL : xU
  end
  return arg1,arg2
end

function cos(x::MC{N}) where N
  Intv = cos(x.Intv)
  xL = x.Intv.lo
  xU = x.Intv.hi
  xLc = Intv.lo
  xUc = Intv.hi
  eps_max,eps_min = cos_arg(x.Intv.lo,x.Intv.hi)
  midcc,cc_id = mid3(x.cc,x.cv,eps_max)
  midcv,cv_id = mid3(x.cc,x.cv,eps_min)
  cc,dcc = cc_cos(midcc,x.Intv.lo,x.Intv.hi)
  cv,dcv = cv_cos(midcv,x.Intv.lo,x.Intv.hi)
  if (MC_param.mu>=1)
    gcc1,gdcc1 = cc_cos(x.cv,x.Intv.lo,x.Intv.hi)
    gcv1,gdcv1 = cv_cos(x.cv,x.Intv.lo,x.Intv.hi)
    gcc2,gdcc2 = cc_cos(x.cc,x.Intv.lo,x.Intv.hi)
    gcv2,gdcv2 = cv_cos(x.cc,x.Intv.lo,x.Intv.hi)
    cv_grad = max(0.0,gdcv1)*x.cv_grad + min(0.0,gdcv2)*x.cc_grad
    cc_grad = min(0.0,gdcc1)*x.cv_grad + max(0.0,gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv,cc,cv_grad,cc_grad = cut(xLc,xUc,cv,cc,cv_grad,cc_grad)
  end
  return MC{N}(cv, cc, Intv, cv_grad, cc_grad, x.cnst)
end

sin(x::MC) = cos(x-pi/2.0)

# pivot point calculation function for convex relaxation of complimentary error function
tan_env(x,y,z) = (x-y)-(tan(x)-tan(y))/(1.0+tan(x)^2)
# derivative of pivot point calculation function for convex relaxation of complimentary error function
tan_envd(x,y,z)= 2*tan(x)/(1.0+tan(x)^2)*(tan(x)-tan(y))

# convex relaxation (envelope) of tangent function
function cv_tan(x,xL,xU)
  p = 0.0
  if (xL>=0.0)
    return tan(x),sec(x)^2
  elseif (xU<=0.0)
    return line_seg(x,xL,tan(xL),xU,tan(xU)),dline_seg(x,xL,tan(xL),xU,tan(xU),sec(x)^2)
  else
    try
      p = secant(0.0,xU,0.0,xU,tan_env,xL,0.0)
    catch e
      if isa(e, ErrorException)
        p = golden_section(0.0,xU,tan_env,xL,0.0)
      end
    end
    if (x<=p)
      return line_seg(x,xL,tan(xL),p,tan(p)),dline_seg(x,xL,tan(xL),p,tan(p),sec(x)^2)
    else
      return tan(x),sec(x)^2
    end
  end
end
# concave relaxation (envelope) of tangent function
function cc_tan(x,xL,xU)
  p = 0.0
  if (xL>=0.0)
    return line_seg(x,xL,tan(xL),xU,tan(xU)),dline_seg(x,xL,tan(xL),xU,tan(xU),sec(x)^2)
  elseif (xU<=0.0)
    return tan(x),sec(x)^2
  else
    try
      p = secant(0.0,xL,xL,0.0,tan_env,xU,0.0)
    catch e
      if isa(e, ErrorException)
        p = golden_section(xL,0.0,tan_env,xU,0.0)
      end
    end
    if (x<=p)
       return tan(x),sec(x)^2
    else
       return line_seg(x,p,tan(p),xU,tan(xU)),dline_seg(x,p,tan(p),xU,tan(xU),sec(x)^2)
     end
  end
end
function tan(x::MC{N}) where N
  Intv = tan(x.Intv)
  xL = x.Intv.lo
  xU = x.Intv.hi
  xLc = Intv.lo
  xUc = Intv.hi
  eps_max = x.Intv.hi
  eps_min = x.Intv.lo
  midcc,cc_id = mid3(x.cc,x.cv,eps_max)
  midcv,cv_id = mid3(x.cc,x.cv,eps_min)
  cc,dcc = cc_tan(midcc,x.Intv.lo,x.Intv.hi)
  cv,dcv = cv_tan(midcv,x.Intv.lo,x.Intv.hi)
  if ((Intv.lo==-Inf)||(Intv.hi==Inf))
    error("Function unbounded on domain")
  end
  if (MC_param.mu >= 1)
    gcc1,gdcc1 = cc_tan(x.cv,x.Intv.lo,x.Intv.hi)
    gcv1,gdcv1 = cv_tan(x.cv,x.Intv.lo,x.Intv.hi)
    gcc2,gdcc2 = cc_tan(x.cc,x.Intv.lo,x.Intv.hi)
    gcv2,gdcv2 = cv_tan(x.cc,x.Intv.lo,x.Intv.hi)
    cv_grad = max(0.0,gdcv1)*x.cv_grad + min(0.0,gdcv2)*x.cc_grad
    cc_grad = min(0.0,gdcc1)*x.cv_grad + max(0.0,gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv,cc,cv_grad,cc_grad = cut(xLc,xUc,cv,cc,cv_grad,cc_grad)
  end
  return MC{N}(cv, cc, Intv, cv_grad, cc_grad, x.cnst)
end

deg2rad(x::MC) = DegToRadIntv*x

sec(x::MC)= inv(sin(x))
csc(x::MC)= inv(cos(x))
cot(x::MC)= inv(tan(x))

asec(x::MC) = acos(inv(x))
acsc(x::MC) = asin(inv(x))
acot(x::MC) = atan(inv(x))

csch(x::MC) = inv(sinh(x))
coth(x::MC) = inv(tanh(x))

acsch(x::MC) = log(sqrt(1.0+inv(sqr(x)))+inv(x))
acoth(x::MC) = 0.5*(log(1.0+inv(x))-log(1.0-inv(x)))

sind(x::MC) = sin(deg2rad(x))
cosd(x::MC) = cos(deg2rad(x))
secd(x::MC) = inv(cosd(x))
cscd(x::MC) = inv(sind(x))
cotd(x::MC) = inv(tand(x))

asind(x::MC) = asin(deg2rad(x))
acosd(x::MC) = acos(deg2rad(x))
atand(x::MC) = atan(deg2rad(x))
asecd(x::MC) = asec(deg2rad(x))
acscd(x::MC) = acsc(deg2rad(x))
acotd(x::MC) = acot(deg2rad(x))
