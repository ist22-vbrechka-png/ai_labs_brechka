clc; clear; close all;

%% --- Крок 1. Визначення зображень букв (7x5) ---

% Буква В (без змін)
V = [ 1,  1,  1, -1, -1;
      1, -1, -1,  1, -1;
      1, -1, -1,  1, -1;
      1,  1,  1, -1, -1;
      1, -1, -1,  1, -1;
      1, -1, -1,  1, -1;
      1,  1,  1, -1, -1];

% Буква Л (нова)
L = [-1, -1,  1, -1, -1;
     -1,  1, -1,  1, -1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1];

% Буква А (без змін)
A = [-1, -1,  1, -1, -1;
     -1,  1, -1,  1, -1;
      1, -1, -1, -1,  1;
      1,  1,  1,  1,  1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1];

% Буква Д (нова)
D = [-1,  1,  1,  1, -1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1;
      1, -1, -1, -1,  1;
     -1,  1,  1,  1, -1];

% Вектори
x1 = V(:);
x2 = L(:);
x3 = A(:);
x4 = D(:);

patterns = [x1, x2, x3, x4];

% Цільові вектори (залишаємо ту ж логіку)
targets  = [  1 -1 -1 -1;
             -1  1 -1 -1;
             -1 -1  1 -1;
             -1 -1 -1  1]';

n = size(patterns, 1);
m = 4;

%% --- Алгоритм Хебба ---
W = zeros(m, n+1);

max_epochs = 100;
converged  = false;

for epoch = 1:max_epochs
    for k = 1:4
        xk = [1; patterns(:,k)];
        yk = targets(:,k);

        S     = W * xk;
        y_out = sign(S);
        y_out(y_out == 0) = -1;

        if any(y_out ~= yk)
            for i = 1:m
                W(i,:) = W(i,:) + yk(i) * xk';
            end
        end
    end

    % перевірка
    all_correct = true;
    for k = 1:4
        xk    = [1; patterns(:,k)];
        y_out = sign(W * xk);
        y_out(y_out == 0) = -1;

        if any(y_out ~= targets(:,k))
            all_correct = false;
            break;
        end
    end

    if all_correct
        fprintf('Навчання завершено за %d епох\n', epoch);
        converged = true;
        break;
    end
end

if ~converged
    fprintf('Не збіглося!\n');
end

%% --- Тест ---
letter_names = {'В', 'Л', 'А', 'Д'};

fprintf('\n=== Тест ===\n');
for k = 1:4
    xk = [1; patterns(:,k)];
    y_out = sign(W * xk);
    y_out(y_out == 0) = -1;

    [~, idx] = max(y_out);

    fprintf('Вхід: %s -> %s\n', ...
        letter_names{k}, letter_names{idx});
end

%% --- Візуалізація ---
figure;
for i = 1:4
    subplot(1,4,i);
    imagesc(reshape(patterns(:,i),7,5));
    colormap(gray); axis off;
    title(letter_names{i});
end
