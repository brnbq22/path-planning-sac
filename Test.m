close all;
clearvars;
clc;
%% Các tham số Train
max_episode = 1000;     % Số lượt tối đa
max_step = 2000;        % Số bước tối đa trong 1 lượt
frequency = 1;          % Tần suất cập nhật
buffer_size = 100000;   % Kích thước tối đa của Buffer
batch_size = 256;       % Kích thước Batch
%% Nhập Agent và Buffer
load('Trained.mat');
speed = zeros(1, max_episode);
% %% Tạo đối tượng VideoWriter với độ phân giải cao nhất
% v = VideoWriter('Video 19 19.avi');
% v.Quality = 100; % Chất lượng cao nhất
% v.FrameRate = 30; % Số khung hình trên giây
% %% Tạo figure mới với kích thước lớn hơn
% figure('Position', [0, 0, 1080, 1080]); % Kích thước figure
% %% Mở VideoWriter để ghi
% open(v);
%% Test
for i = 1:max_episode
    %% Reset trạng thái
    path = 0;
    position = [1; 1; pi/4];
    lidarData = env.readLidar(position);
    U_rep = env.repulsion(lidarData);
    p_g = (env.Goal(1:2) - position(1:2))./[env.Limx(2); env.Limy(2)];
    state = [position./[env.Limx(2); env.Limy(2); pi]; p_g; U_rep];
    x = position(1);
    y = position(2);
    gamma = 1;
    score = 0;
    entropy = 0;
    for j = 1:max_step
        %% Chọn hành động và thực hiện hành động
        [action, logProb] = agent.selectAction(state);
        [nextState, reward, isDone] = env.step(state, action);
        %% Tính Critic và Target để kiểm tra
        if j == 1
            [Q1, ~] = agent.criticForward(agent.CriticQ1Weights, state, action);
            [Q2, ~] = agent.criticForward(agent.CriticQ2Weights, state, action);
            [T1, ~] = agent.criticForward(agent.TargetQ1Weights, state, action);
            [T2, ~] = agent.criticForward(agent.TargetQ2Weights, state, action);
        end
        %% Vẽ
        path = path + norm(nextState(1:2).*[env.Limx(2); env.Limy(2)] - state(1:2).*[env.Limx(2); env.Limy(2)]);
        x = cat(2, x, nextState(1)*env.Limx(2));
        y = cat(2, y, nextState(2)*env.Limy(2));
        % env.plot(state, x, y);
        % drawnow;
        % frame = getframe(gcf);
        % pause(0.01);
        % writeVideo(v, frame);
        %% Chuẩn bị sang trạng thái mới
        score = score + gamma*reward;
        entropy = entropy - gamma*logProb;
        gamma = gamma*agent.Gamma;
        buffer_count = buffer_count + 1;
        state = nextState;
        %% Kiểm tra điều kiện dừng
        if isDone == 1
            break;
        end
    end
    %% Tính tốc độ
    speed(:, i) = path/(j*0.1);
    %% Hiển thị
    fprintf('Lượt thứ: %-6d Số bước: %-5d Tổng điểm: %-8.2f Entropy: %-8.2f Alpha: %-8.4f Q1: %-8.2f Q2: %-8.2f T1: %-8.2f T2: %-8.2f Tốc độ %.2f (m/s)\n', ...
             i, j, score, entropy, agent.Alpha, Q1, Q2, T1, T2, speed(i));
end
% %% Đóng đối tượng VideoWriter
% close(v);
%% Tốc độ trung bình
speedMean = mean(speed);
speedMax = max(speed);
speedMin = min(speed);
%% Vẽ Map
figure(1);
env.plotMap;
%% Vẽ Path
figure(2);
env.plotPath(x, y);
%% Vẽ đồ thị điểm thưởng trung bình
figure(3);
rewardAverage = zeros(size(rewardSave));
for e = 1:size(rewardSave, 2)
    rewardAverage(e) = mean(rewardSave(max(1, e - 200):e));
end
plot(rewardSave, 'b');
hold on;
plot(rewardAverage, 'r', 'LineWidth', 3);
grid on;
xlim([0, size(rewardSave, 2)]);
ylim([0, 1200]);
legend('Điểm thưởng', 'Điểm thưởng trung bình');
xlabel('Số lượt huấn luyện');
ylabel('Điểm thưởng');
title('Đồ thị điểm thưởng và điểm thưởng trung bình');
saveas(gcf, 'Reward 19 19.png'); % Lưu dưới dạng PNG