close all;
clearvars;
clc;
%% Các tham số Train
max_episode = 1000;     % Số lượt tối đa
max_step = 2000;        % Số bước tối đa trong 1 lượt
frequency = 1;          % Tần suất cập nhật
buffer_size = 1000000;  % Kích thước tối đa của Buffer
batch_size = 256;       % Kích thước Batch
%% Khởi tạo Environment
env = Environment([0; 20], ...      % Giới hạn trục x của map
                  [0; 20], ...      % Giới hạn trục y của map
                  [19; 19; 1], ...  % Điểm đích (tâm và bán kính)
                  1, ...            % Có vật cản không (1 có, 0 không)
                  [4, 16, 2; ...
                   14, 12, 2; ...
                   14, 2, 1; ...
                   12, 8, 1; ...
                   8, 12, 2; ...
                   10, 4, 2; ...
                   14, 18, 1; ...
                   17, 8, 1; ...
                   3, 10, 1; ...
                   18, 16, 1; ...
                   18, 4, 2; ...
                   10, 18, 2; ...
                   6, 2, 1; ...
                   3, 6, 1; ...
                   6, 6, 1], ...    % Vật cản tròn (tâm và bán kính)
                  31, ...           % Số tia Lidar
                  5);               % Tầm xa của Lidar
%% Khởi tạo Agent
agent = Agent(3 + 2 + env.Num_rays, ...  % Kích thước trạng thái
              2, ...                     % Kích thước hành động
              32, ...                    % Kích thước lớp Fully Connected
              0.001, ...                 % Tốc độ học cho Actor
              0.001, ...                 % Tốc độ học cho Critric
              0.0003, ...                % Tốc độ học cho Entropy
              0.99, ...                  % Hệ số chiết khấu
              0.005); ...                % Hệ số cập nhật Target
%% Nhập Agent và Buffer
% load('Trained.mat');
%% Khởi tạo 1 số thứ còn lại
buffer = zeros(agent.StateSize*2 + agent.ActionSize + 2, buffer_size);  % Buffer toàn số 0
buffer_count = 1;                                                       % Đếm Buffer
rewardSave = zeros(1, max_episode);                                     % Lưu điểm thưởng
%% Train
for i = 1:max_episode
    %% Reset trạng thái
    position = [1; 1; pi/4];
    lidarData = env.readLidar(position);
    p_g = (env.Goal(1:2) - position(1:2))./[env.Limx(2); env.Limy(2)];
    state = [position./[env.Limx(2); env.Limy(2); pi]; p_g; lidarData/env.Max_distance];
    gamma = 1;
    score = 0;
    entropy = 0;
    for j = 1:max_step
        %% Chọn hành động và thực hiện hành động
        [action, logProb] = agent.selectAction(state);
        [nextState, reward, isDone] = env.step(state, action);
        buffer(:, max(1, mod(buffer_count, buffer_size))) = [state; action; reward; nextState; isDone];
        %% Cập nhật các trọng số
        if buffer_count > batch_size && mod(j, frequency) == 0
            % Lấy Batch
            randomIndices = randperm(min(buffer_size, buffer_count), batch_size);
            batch = buffer(:, randomIndices);
            % Cập nhật
            agent = agent.updateCriticQ1(batch);
            agent = agent.updateCriticQ2(batch);
            agent = agent.updateActor(batch);
            agent = agent.updateTemperature(batch);
            agent = agent.updateTargetQ1();
            agent = agent.updateTargetQ2();
        end
        %% Tính Critic và Target để kiểm tra
        if j == 1
            [Q1, ~] = agent.criticForward(agent.CriticQ1Weights, state, action);
            [Q2, ~] = agent.criticForward(agent.CriticQ2Weights, state, action);
            [T1, ~] = agent.criticForward(agent.TargetQ1Weights, state, action);
            [T2, ~] = agent.criticForward(agent.TargetQ2Weights, state, action);
        end
        %% Vẽ
        % env.plot(state);
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
    %% Lưu các thứ
    rewardSave(i) = score;
    save('Trained.mat', 'env', 'agent', 'buffer', 'buffer_count', 'rewardSave');
    %% Hiển thị
    fprintf('Lượt thứ: %-6d Số bước: %-5d Tổng số bước: %-7d Tổng điểm: %-8.2f Entropy: %-8.2f Alpha: %-8.4f Q1: %-8.2f Q2: %-8.2f T1: %-8.2f T2: %-8.2f\n', ...
             i, j, buffer_count - 1, score, entropy, agent.Alpha, Q1, Q2, T1, T2);
end