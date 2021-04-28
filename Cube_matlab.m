%% setup 
clc; close all; clear all;
addpath 'D:\\RPI\\21 spring\\FEP\\Project\\fem3DCubeTet\\fem3DCubeTet'
%%
model = createpde('structural','static-solid');
importGeometry(model,'3D_cube.stl');
mesh_default = generateMesh(model,'GeometricOrder','linear','Hgrad',1);     % Hgrad -> growth rate in mesh, =1 no growth, i.e constant

figure
pdemesh(mesh_default,'NodeLabels','on')
title('FEP: Tetrahedron Element Mesh on Solid Cube')

mesh_Hmax = generateMesh(model,'Hmax',20,'GeometricOrder','linear','Hgrad',1)  % 20 height of the cube
figure
pdemesh(mesh_Hmax,'NodeLabels','on')
title('FEP: N546-Tetrahedron Element Mesh on Solid Cube')
%% Save the variables
Nodes = mesh_Hmax.Nodes;
Elements = mesh_Hmax.Elements;
MaxElementSize = mesh_Hmax.MaxElementSize;
MinElementSize = mesh_Hmax.MinElementSize;
MeshGradation = mesh_Hmax.MeshGradation;
GeometricOrder = mesh_Hmax.GeometricOrder;
Nodes_f2 = findNodes(mesh_Hmax,'region','Face',2);
Nodes_f6 = findNodes(mesh_Hmax,'region','Face',6);

%% visualize the mesh
Nf2 = findNodes(mesh_Hmax,'region','Face',2);       % face 6 is the top face (for force); face 2 is the bottom face (for constraint)
figure
pdemesh(model,'NodeLabels','on')
hold on
plot(mesh_Hmax(1,Nf2),mesh_Hmax.Nodes(2,Nf2),'ok','MarkerFaceColor','g')  

%tet_mesh_display(mesh_Hmax.Nodes)

%% Deformed cube
Mesh = [14,46,154,880];    %number of nodes in different mesh
Mesh_size = [];
h = 20;                     % Largest mesh size
for i = 1:4
    Mesh_size(i) = h ;
    h = h/2;
end 
L2_error = zeros(1,4);
for m = 1:4
    mesh_data = sprintf('Cube3D_tet_N%d_April24.mat',Mesh(m));
    load(mesh_data);
    defo_data = sprintf('x_N%d.csv',Mesh(m));
    NodalDefo = load(defo_data);      % Nodal_deformation
    M = size(NodalDefo,1)/3;
    DefoNodes = zeros(size(Nodes,1), size(Nodes,2));
    k = 1;
    for i = 1:M
        for j = 1:3
            DefoNodes(j,i) = NodalDefo(k); % convert to array form
            k = k+1;
        end
    end
    % plotting the deformed cube
    P = DefoNodes';
    Q = Nodes';
    q = boundary(P);
    l = boundary(Q);
    plot3(P(:,1),P(:,2),P(:,3),'.','MarkerSize',20,'color', 'g')
    hold on
    trisurf(q,P(:,1),P(:,2),P(:,3),'Facecolor','red','FaceAlpha',0.1)
    hold on
    plot3(Q(:,1),Q(:,2),Q(:,3),'.','MarkerSize',20,'color', 'r')
    hold on
    trisurf(l,Q(:,1),Q(:,2),Q(:,3),'Facecolor','green','FaceAlpha',0.1)
    % Find the displacement for the compression force
    F = 150; E = 210e9;
    A = 400; % sq.meter
    P = F/A;
    pair = [];  deform_analy = [];nu = 0.3;
    dist_Z = []; % to store the difference in distance in z direction
    for i = 1: size(Nodes,2)
        dist_Z = abs(Nodes(3,2) - Nodes(3,i));
        deform_analy(3,i) = P/E*(dist_Z);
        deform_analy(1,i) = nu*P/E*(dist_Z);
        deform_analy(2,i) = nu*P/E*(dist_Z);
    end
    %L2 norm of the error
    err_sq = 0;
    sum_sq_error = 0;
    for i = 1:size(Nodes,2)
        for j = 1:3
            err_sq = (deform_analy(j,i) - DefoNodes(j,i))^2;
            sum_sq_error = sum_sq_error + err_sq;
        end
    end
    L2_error(m) = sqrt(sum_sq_error); 
end
% log/log plot
figure()
plot(log(flip(Mesh_size)),log(L2_error),'LineWidth',2);
title('log/log plot of error vs mesh size')
xlabel('Mesh size (h)')
ylabel('log (L2 error)')

