% ============================================================
% ЛАБОРАТОРНА РОБОТА 9 — Варіант 4
% Моделювання об'єктів управління на основі нейронних мереж
%
% y(x) = 7 - sin(5*pi/2) - cos(x)
% z(x,y) = (x+2)^2 / (1 + y^2)
% ============================================================

%% --- Конфігурації мереж ---
configs = {
    'feedforwardnet',   [10],     'Feed-forward, 1 шар, 10 нейронів';
    'feedforwardnet',   [20],     'Feed-forward, 1 шар, 20 нейронів';
    'cascadeforwardnet',[20],     'Cascade-forward, 1 шар, 20 нейронів';
    'cascadeforwardnet',[10 10],  'Cascade-forward, 2 шари по 10 нейронів';
    'elmannet',         [15],     'Elman, 1 шар, 15 нейронів';
    'elmannet',         [5 5 5],  'Elman, 3 шари по 5 нейронів';
};

short_labels = {'FF 1x10','FF 1x20','CF 1x20','CF 2x10','EL 1x15','EL 3x5'};

%% Допоміжні функції
function net = create_net(net_type, layers)
    switch net_type
        case 'feedforwardnet',    net = feedforwardnet(layers);
        case 'cascadeforwardnet', net = cascadeforwardnet(layers);
        case 'elmannet',          net = elmannet(layers);
    end
    net.trainParam.epochs     = 1000;
    net.trainParam.goal       = 1e-6;
    net.trainParam.showWindow = false;
end

function print_table(configs, results, func_name)
    fprintf('\n--- Зведена таблиця: %s ---\n', func_name);
    fprintf('%-45s | Похибка (%%)\n', 'Конфігурація мережі');
    fprintf('%s\n', repmat('-',1,60));
    for i = 1:length(results)
        marker = '';
        if results(i) == min(results), marker = ' <- найкраща'; end
        if results(i) == max(results), marker = ' <- найгірша'; end
        fprintf('%-45s | %.6f%s\n', configs{i,3}, results(i), marker);
    end
    fprintf('%s\n', repmat('=',1,60));
end

%% ====================== ЧАСТИНА 1: z1(x,y) ======================
fprintf('==========================================================\n');
fprintf('   ЧАСТИНА 1: z1(x,y) = (x+2)^2 / (1+y^2)              \n');
fprintf('==========================================================\n');

N = 500;
x1 = linspace(0, pi/2, N);
y1 = linspace(0, pi/2, N);
Input1  = [x1; y1];
Output1 = (x1 + 2).^2 ./ (1 + y1.^2);

results1 = zeros(1,6);
nets1    = cell(1,6);

for i = 1:6
    fprintf('\n=== %s ===\n', configs{i,3});
    net = create_net(configs{i,1}, configs{i,2});
    [net, tr] = train(net, Input1, Output1);
    Out_nn = net(Input1);
    err = mean(abs(Output1 - Out_nn) ./ (abs(Output1) + 1e-10)) * 100;
    results1(i) = err;
    nets1{i}    = net;
    fprintf('Середня відносна похибка: %.6f %%\n', err);
    fprintf('MSE: %.2e\n', tr.best_perf);
end

print_table(configs, results1, 'z1(x,y) = (x+2)^2/(1+y^2)');

%% ====================== ЧАСТИНА 2: y(x) ======================
fprintf('\n==========================================================\n');
fprintf('   ЧАСТИНА 2: y(x) = 7 - sin(5*pi/2) - cos(x)          \n');
fprintf('==========================================================\n');

% Примітка: sin(5*pi/2) = 1 => y(x) = 7 - 1 - cos(x) = 6 - cos(x)
% Але залишаємо точну формулу для повноти

x2 = linspace(0, pi, N);
Input2  = [x2; zeros(1,N)];
Output2 = 7 - sin(5*pi/2) - cos(x2);   % = 6 - cos(x)

results2 = zeros(1,6);
nets2    = cell(1,6);

for i = 1:6
    fprintf('\n=== %s ===\n', configs{i,3});
    net = create_net(configs{i,1}, configs{i,2});
    [net, tr] = train(net, Input2, Output2);
    Out_nn = net(Input2);
    err = mean(abs(Output2 - Out_nn) ./ (abs(Output2) + 1e-10)) * 100;
    results2(i) = err;
    nets2{i}    = net;
    fprintf('Середня відносна похибка: %.6f %%\n', err);
    fprintf('MSE: %.2e\n', tr.best_perf);
end

print_table(configs, results2, 'y(x) = 7 - sin(5pi/2) - cos(x)');

%% ====================== ГРАФІКИ ======================

%% Графік 1: Порівняння похибок (grouped bar)
figure('Name','Порівняння похибок — Варіант 4','Position',[100 100 900 500]);
b = bar([results1' results2'], 'grouped');
b(1).FaceColor = [0.2 0.6 0.9];
b(2).FaceColor = [0.9 0.4 0.2];
set(gca,'XTickLabel',short_labels,'FontSize',11);
ylabel('Середня відносна похибка (%)');
title('Варіант 4 — Порівняння точності мереж');
legend({'z1(x,y) = (x+2)^2/(1+y^2)', 'y(x) = 7-sin(5\pi/2)-cos(x)'}, ...
       'Location','northwest');
grid on;

%% Графік 2: Детальна апроксимація — z1(x,y)
figure('Name','Детальна апроксимація — z1','Position',[100 100 1200 700]);
sgtitle('Варіант 4 — z1(x,y) = (x+2)^2/(1+y^2)','FontSize',14);
for i = 1:6
    subplot(3,2,i);
    plot(x1, Output1, 'b-', 'LineWidth', 2); hold on;
    plot(x1, nets1{i}(Input1), 'r--', 'LineWidth', 1.5);
    xlabel('x (y=x)'); ylabel('z1');
    title(sprintf('%s  (%.4f%%)', short_labels{i}, results1(i)));
    legend('Еталон','НМ','Location','best');
    grid on;
end

%% Графік 3: Детальна апроксимація — y(x)
figure('Name','Детальна апроксимація — y(x)','Position',[100 100 1200 700]);
sgtitle('Варіант 4 — y(x) = 7 - sin(5\pi/2) - cos(x)','FontSize',14);
for i = 1:6
    subplot(3,2,i);
    plot(x2, Output2, 'b-', 'LineWidth', 2); hold on;
    plot(x2, nets2{i}(Input2), 'r--', 'LineWidth', 1.5);
    xlabel('x'); ylabel('y(x)');
    title(sprintf('%s  (%.4f%%)', short_labels{i}, results2(i)));
    legend('Еталон','НМ','Location','best');
    grid on;
end

%% Графік 4: 3D поверхня z1 — еталон vs краща мережа
[~, best1] = min(results1);
Ng = 40;
[xg,yg] = meshgrid(linspace(0,pi/2,Ng));
Zg_true = (xg + 2).^2 ./ (1 + yg.^2);
Zg_nn   = reshape(nets1{best1}([xg(:)'; yg(:)']), Ng, Ng);

figure('Name','3D Поверхня z1 (Var4)','Position',[100 100 1100 500]);
subplot(1,2,1);
surf(xg,yg,Zg_true,'EdgeColor','none'); title('Еталон'); colorbar;
xlabel('x'); ylabel('y'); zlabel('z1');
subplot(1,2,2);
surf(xg,yg,Zg_nn,'EdgeColor','none');
title(sprintf('НМ: %s  (%.4f%%)', short_labels{best1}, results1(best1))); colorbar;
xlabel('x'); ylabel('y'); zlabel('z1');
sgtitle('Варіант 4 — 3D поверхня z1(x,y) = (x+2)^2/(1+y^2)');