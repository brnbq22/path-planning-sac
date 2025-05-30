classdef Environment
    %% Định nghĩa các thuộc tính
    properties
        Limx          % Giới hạn trục x
        Limy          % Giới hạn trục x
        Goal          % Tâm đích và bán kính
        IsObs         % Có vật cản không (true, false)
        Obstacles     % Vật cản tròn
        Num_rays      % Số tia Lidar
        Max_distance  % Tầm xa của Lidar
        RobotRadius   % Bán kính Robot
        d             % Khoảng cách giữa 2 bánh
        dt            % Thời gian trích mẫu
    end
    %% Định nghĩa các phương thức
    methods
        %% Phương thức để khởi tạo
        function obj = Environment(limx, limy, goal, isObs, obstacles, num_rays, max_distance)
            if nargin > 0
                obj.Limx = limx;
                obj.Limy = limy;
                obj.Goal = goal;
                obj.IsObs = isObs;
                obj.Obstacles = obstacles;
                obj.Num_rays = num_rays;
                obj.Max_distance = max_distance;
                obj.RobotRadius = 0.25;
                obj.d = 0.37;
                obj.dt = 0.1;
            end
        end
        %% Hàm chuyển trạng thái của môi trường
        function [nextState, reward, isDone] = step(obj, state, action)
            % Lấy trạng thái và hành động
            x = state(1)*obj.Limx(2);
            y = state(2)*obj.Limy(2);
            theta = state(3)*pi;
            u1 = action(1);
            u2 = action(2);
            % Động học của chiếc xe
            x1 = x + obj.dt*(0.5*(u1 + u2)*cos(theta));
            y1 = y + obj.dt*(0.5*(u1 + u2)*sin(theta));
            theta1 = theta + obj.dt*((u1 - u2)*obj.d);
            % Kiểm tra xem có bị đâm không
            lidarData1 = obj.readLidar([x1; y1; theta1]);
            isCollision = false;
            if min(lidarData1) < obj.RobotRadius
                isCollision = true;
            end
            % Tính khoảng cách so với đích
            oldDistanceToGoal = norm([x - obj.Goal(1); y - obj.Goal(2)]);
            newDistanceToGoal = norm([x1 - obj.Goal(1); y1 - obj.Goal(2)]);
            % Tính cos bằng tích vô hướng
            a = [cos(theta1); sin(theta1)];
            b = obj.Goal(1:2) - [x1; y1];
            cosAlpha = a'*b/norm(b);
            % Kiểm tra điều kiện và tính điểm thưởng
            if isCollision
                reward = -1000;
                theta1 = theta1 + pi/2;
                isDone = 0;
            elseif newDistanceToGoal < obj.Goal(3)
                reward = 5000;
                isDone = 1;
            else
                reward = 50*(oldDistanceToGoal - newDistanceToGoal) + 5*cosAlpha;
                isDone = 0;
            end
            % Cập nhật trạng thái mới
            position1 = [x1; y1; theta1];
            p_g1 = (obj.Goal(1:2) - position1(1:2))./[obj.Limx(2); obj.Limy(2)];
            nextState = [position1./[obj.Limx(2); obj.Limy(2); pi]; p_g1; lidarData1/obj.Max_distance];
        end
        %% Tính U_rep
        function U_rep = repulsion(obj, lidarData)
            U_rep = (1./lidarData - 1/obj.Max_distance).^2;
        end
        %% Đọc Lidar
        function lidarData = readLidar(obj, state)
            % Tính toán giá trị LIDAR
            lidarData = ones(obj.Num_rays, 1) * obj.Max_distance;
            angles = linspace(-pi/2, pi/2, obj.Num_rays);
            % Vòng lặp đọc từng tia LIDAR
            for i = 1:obj.Num_rays
                % Phương trình đường thẳng của tia LIDAR
                angle = angles(i) + state(3);
                dx = cos(angle);
                dy = sin(angle); 
                % Tìm giao điểm với hình tròn
                if obj.IsObs == 1
                    for j = 1:size(obj.Obstacles, 1)
                        xc = obj.Obstacles(j, 1);
                        yc = obj.Obstacles(j, 2);
                        r = obj.Obstacles(j, 3);
                        % Giải phương trình để tìm giao điểm
                        a = dx^2 + dy^2;
                        b = 2*(dx*(state(1) - xc) + dy*(state(2) - yc));
                        c = (state(1) - xc)^2 + (state(2) - yc)^2 - r^2;
                        delta = b^2 - 4*a*c;
                        % Nếu phương trình giao điểm có nghiệm
                        if delta >= 0
                            t1 = (-b + sqrt(delta))/(2*a);
                            t2 = (-b - sqrt(delta))/(2*a);
                            if t1 > 0 && t1 <= obj.Max_distance
                                lidarData(i) = min(lidarData(i), t1);
                            end
                            if t2 > 0 && t2 <= obj.Max_distance
                                lidarData(i) = min(lidarData(i), t2);
                            end
                        end
                    end
                end
                % Tìm giao điểm với khung viền môi trường
                environment_bounds = [obj.Limx(1), obj.Limy(1), obj.Limx(2) - obj.Limx(1), obj.Limy(2) - obj.Limy(1)];
                x_min = environment_bounds(1);
                y_min = environment_bounds(2);
                x_max = x_min + environment_bounds(3);
                y_max = y_min + environment_bounds(4);
                % Khởi tạo ma trận
                t_values = [];
                % Tìm giao điểm
                if dx ~= 0
                    t1 = (x_min - state(1)) / dx;
                    y1 = state(2) + t1 * dy;
                    if y1 >= y_min && y1 <= y_max && t1 > 0 && t1 <= obj.Max_distance
                        t_values = cat(2, t_values, t1);
                    end 
                    t2 = (x_max - state(1)) / dx;
                    y2 = state(2) + t2 * dy;
                    if y2 >= y_min && y2 <= y_max && t2 > 0 && t2 <= obj.Max_distance
                        t_values = cat(2, t_values, t2);
                    end
                end
                % Tìm giao điểm
                if dy ~= 0
                    t3 = (y_min - state(2)) / dy;
                    x3 = state(1) + t3 * dx;
                    if x3 >= x_min && x3 <= x_max && t3 > 0 && t3 <= obj.Max_distance
                        t_values = cat(2, t_values, t3);
                    end
                    t4 = (y_max - state(2)) / dy;
                    x4 = state(1) + t4 * dx;
                    if x4 >= x_min && x4 <= x_max && t4 > 0 && t4 <= obj.Max_distance
                        t_values = cat(2, t_values, t4);
                    end
                end
                % Kiểm tra xem có giao điểm không
                if ~isempty(t_values)
                    lidarData(i) = min(lidarData(i), min(t_values));
                end
                if lidarData(i) == 0
                    lidarData(i) = obj.Max_distance;
                end
            end
        end
        %% Vẽ
        function plot(obj, state, x, y)
            clf; % Xóa nội dung của figure
            hold on;
            % Tách
            position = state(1:3).*[obj.Limx(2); obj.Limy(2); pi];
            lidarData = obj.readLidar(position);
            % Vẽ xe và path
            plot(x, y, 'LineWidth', 2);
            plot(position(1), position(2), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
            % Vẽ khung
            environment_bounds = [obj.Limx(1), obj.Limy(1), obj.Limx(2) - obj.Limx(1), obj.Limy(2) - obj.Limy(1)];
            rectangle('Position', environment_bounds, 'EdgeColor', 'r', 'LineWidth', 2);
            % Vẽ vật cản
            if obj.IsObs == 1
                for o = 1:size(obj.Obstacles, 1)
                    viscircles(obj.Obstacles(o, 1:2), obj.Obstacles(o, 3), 'Color', 'r', 'LineWidth', 2);
                end
            end
            % Vẽ điểm đầu
            plot(1, 1, 'bx', 'LineWidth', 2);
            % Vẽ đích
            viscircles(obj.Goal(1:2)', obj.Goal(3), 'Color', 'g', 'LineWidth', 2);
            % Vẽ các tia LIDAR
            delete(findall(gcf, 'Type', 'line', 'Color', 'b')); % Xóa các tia LIDAR cũ
            angles = linspace(-pi/2, pi/2, obj.Num_rays);
            for i = 1:obj.Num_rays
                angle = angles(i) + position(3);
                end_x = position(1) + lidarData(i) * cos(angle);
                end_y = position(2) + lidarData(i) * sin(angle);
                plot([position(1), end_x], [position(2), end_y], 'b');
            end
            % Cài đặt
            axis equal;
            xlim(obj.Limx);
            ylim(obj.Limy);
            xlabel('Tọa độ trục x');
            ylabel('Tọa độ trục y');
            title('Robot di động tránh vật cản');
        end
        %% Vẽ Path
        function plotPath(obj, x, y)
            % Vẽ đường đi
            plot(x, y, 'LineWidth', 2);
            hold on;
            % Vẽ khung
            environment_bounds = [obj.Limx(1), obj.Limy(1), obj.Limx(2) - obj.Limx(1), obj.Limy(2) - obj.Limy(1)];
            rectangle('Position', environment_bounds, 'EdgeColor', 'r', 'LineWidth', 2);
            % Vẽ vật cản
            if obj.IsObs == 1
                for o = 1:size(obj.Obstacles, 1)
                    viscircles(obj.Obstacles(o, 1:2), obj.Obstacles(o, 3), 'Color', 'r', 'LineWidth', 2);
                end
            end
            % Vẽ điểm đầu
            plot(1, 1, 'bx', 'LineWidth', 2);
            % Vẽ đích
            viscircles(obj.Goal(1:2)', obj.Goal(3), 'Color', 'g', 'LineWidth', 2);
            % Cài đặt
            xlim(obj.Limx);
            ylim(obj.Limy);
            xlabel('Tọa độ trục x');
            ylabel('Tọa độ trục y');
            title('Đường đi của robot sau khi huấn luyện');
            saveas(gcf, 'Path 19 19.png'); % Lưu dưới dạng PNG
        end
        %% Vẽ Map
        function plotMap(obj)
            % Vẽ khung
            environment_bounds = [obj.Limx(1), obj.Limy(1), obj.Limx(2) - obj.Limx(1), obj.Limy(2) - obj.Limy(1)];
            rectangle('Position', environment_bounds, 'EdgeColor', 'r', 'LineWidth', 2);
            % Vẽ vật cản
            if obj.IsObs == 1
                for o = 1:size(obj.Obstacles, 1)
                    viscircles(obj.Obstacles(o, 1:2), obj.Obstacles(o, 3), 'Color', 'r', 'LineWidth', 2);
                end
            end
            hold on;
            % Vẽ điểm đầu
            plot(1, 1, 'bx', 'LineWidth', 2);
            % Vẽ đích
            viscircles(obj.Goal(1:2)', obj.Goal(3), 'Color', 'g', 'LineWidth', 2);
            % Cài đặt
            xlim(obj.Limx);
            ylim(obj.Limy);
            xlabel('Tọa độ trục x');
            ylabel('Tọa độ trục y');
            title('Bản đồ thử nghiệm cho robot di động');
            saveas(gcf, 'Map 19 19.png'); % Lưu dưới dạng PNG
        end
    end
end