import pygame
import cv2
import mediapipe as mp
import os
import time
import urllib.request
import random

# --- SETTINGS ---
WIDTH, HEIGHT = 800, 600
SNAKE_SIZE = 40  # Slightly smaller for better control
FPS = 8  # Reduced from 12 to make snake slower
MOVEMENT_DELAY = 0.15  # Additional delay between movements (seconds)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Controlled Snake with Gestures 🐍")
font = pygame.font.SysFont("Arial", 24, bold=True)
small_font = pygame.font.SysFont("Arial", 18)
clock = pygame.time.Clock()

_USE_TASKS_API = False

# --- SETUP MEDIAPIPE ---
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

    MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    MODELS_DIR = os.path.join(os.getcwd(), "models")
    MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    _USE_TASKS_API = True
    print("✅ Using MediaPipe Tasks API")

except Exception as e:
    print(f"⚙️ Falling back to classic MediaPipe Hands API: {e}")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )

def open_camera(index=0, width=640, height=480):
    """Try opening the camera using multiple backends."""
    for cam_index in [index, 0, 1]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(cam_index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"✅ Camera opened: index={cam_index}, backend={backend}")
                    return cap
            except:
                try:
                    cap.release()
                except:
                    pass
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print("✅ Camera opened with default settings")
        return cap
    
    return None

cap = open_camera()
if cap is None:
    print("❌ ERROR: No camera found!")
    input("Press Enter to exit...")
    pygame.quit()
    exit()

# --- GAME VARIABLES ---
snake = [[200, 200], [180, 200], [160, 200]]
direction = "RIGHT"
next_direction = "RIGHT"  # Buffer for next direction
food = [random.randrange(0, WIDTH, SNAKE_SIZE), random.randrange(0, HEIGHT, SNAKE_SIZE)]
score = 0
high_score = 0
paused = False
last_gesture_time = 0
gesture_cooldown = 0.5  # Increased cooldown for better accuracy
last_move_time = 0
movement_delay = MOVEMENT_DELAY
current_gesture = "None"
extended_count = 0
gesture_stability_count = 0  # For gesture stability
stable_gesture = "None"
stable_finger_count = 0

def count_extended_fingers(landmarks):
    """Count the number of extended fingers with improved accuracy."""
    extended = 0
    
    # Thumb - improved detection
    if landmarks[4].x > landmarks[3].x and landmarks[4].y < landmarks[3].y + 0.1:
        extended += 1
    
    # Other 4 fingers - more strict comparison
    for tip_id in [8, 12, 16, 20]:
        pip_id = tip_id - 2
        mcp_id = tip_id - 3
        
        # Finger is extended if tip is significantly above PIP
        if landmarks[tip_id].y < landmarks[pip_id].y - 0.05:
            extended += 1
    
    return extended

def is_fist(landmarks):
    """Check if hand is a fist (0 or 1 fingers extended)."""
    return count_extended_fingers(landmarks) <= 1

def is_open_hand(landmarks):
    """Check if hand is open (4+ fingers extended)."""
    return count_extended_fingers(landmarks) >= 4

def get_finger_count_gesture(extended_count):
    """Convert finger count to gesture name and direction."""
    if extended_count <= 1:
        return "Fist", "PAUSE"
    elif extended_count == 2:
        return "2 Fingers", "RIGHT"
    elif extended_count == 3:
        return "3 Fingers", "DOWN"
    elif extended_count == 4:
        return "4 Fingers", "LEFT"
    elif extended_count == 5:
        return "Open Hand", "UP"
    else:
        return f"{extended_count} Fingers", None

print("\n🎮 Game Started! Show your hand to the camera...")
print("\n=== CONTROLS ===")
print("  🤜 Fist (0-1 fingers) = Pause/Resume")
print("  ✌️ 2 Fingers = Move RIGHT")
print("  🤟 3 Fingers = Move DOWN")
print("  🖖 4 Fingers = Move LEFT")
print("  🖐️ Open Hand (5 fingers) = Move UP")
print("\n📌 Tips:")
print("  - Hold gestures steadily for better recognition")
print("  - Keep your hand in frame")
print("  - Make clear, distinct finger counts")
print("\nPress 'Q' in camera window to quit, 'R' to reset\n")

# --- MAIN LOOP ---
running = True
while running:
    screen.fill((25, 25, 25))
    current_time = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read camera frame
    success, frame = cap.read()
    if not success:
        print("Camera disconnected! Reconnecting...")
        cap.release()
        time.sleep(1)
        cap = open_camera()
        if cap is None:
            break
        continue

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    fx, fy = None, None
    current_gesture = "None"
    extended_count = 0

    # --- HAND DETECTION ---
    if _USE_TASKS_API:
        try:
            mp_image = Image(ImageFormat.SRGB, rgb_frame)
            timestamp_ms = int(current_time * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                itip = hand[8]
                fx, fy = int(itip.x * WIDTH), int(itip.y * HEIGHT)
                
                # Count extended fingers
                extended_count = 0
                # Thumb
                if hand[4].x > hand[3].x and hand[4].y < hand[3].y + 0.1:
                    extended_count += 1
                # Fingers
                for tip_id in [8, 12, 16, 20]:
                    if hand[tip_id].y < hand[tip_id - 2].y - 0.05:
                        extended_count += 1
                
                current_gesture, _ = get_finger_count_gesture(extended_count)
        except Exception as e:
            _USE_TASKS_API = False

    else:
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            
            # Get finger position for tracking
            fx = int(hand.landmark[8].x * WIDTH)
            fy = int(hand.landmark[8].y * HEIGHT)
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Count extended fingers
            extended_count = count_extended_fingers(hand.landmark)
            current_gesture, _ = get_finger_count_gesture(extended_count)
            
            # Display on camera feed
            cv2.putText(frame, f"Fingers: {extended_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Color code based on gesture
            if extended_count <= 1:
                color = (0, 0, 255)  # Red for fist
            elif extended_count == 2:
                color = (0, 255, 0)  # Green for right
            elif extended_count == 3:
                color = (255, 0, 0)  # Blue for down
            elif extended_count == 4:
                color = (0, 255, 255)  # Yellow for left
            elif extended_count == 5:
                color = (255, 0, 255)  # Magenta for up
            
            cv2.putText(frame, f"Gesture: {current_gesture}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- GESTURE STABILITY (prevent flickering) ---
    if current_gesture != "None":
        if current_gesture == stable_gesture:
            gesture_stability_count += 1
        else:
            gesture_stability_count = 0
            stable_gesture = current_gesture
            stable_finger_count = extended_count
    else:
        gesture_stability_count = 0
    
    # Only act on stable gestures (held for multiple frames)
    gesture_stable = gesture_stability_count > 5

    # --- GESTURE ACTIONS (with improved cooldown) ---
    if gesture_stable and current_time - last_gesture_time > gesture_cooldown:
        
        # Fist = Pause/Resume toggle
        if extended_count <= 1:
            paused = not paused
            last_gesture_time = current_time
            print(f"{'⏸️ Paused' if paused else '▶️ Resumed'} - Fist detected")
        
        # 5 fingers (Open Hand) = Reset game when paused or game over
        elif extended_count == 5 and paused:
            snake = [[200, 200], [180, 200], [160, 200]]
            next_direction = "RIGHT"
            direction = "RIGHT"
            score = 0
            paused = False
            last_gesture_time = current_time
            print("🔄 Game reset!")

    # --- CONTROL SNAKE DIRECTION (only when not paused) ---
    if not paused and gesture_stable:
        # Set direction based on finger count with opposite movement prevention
        if extended_count == 2:  # 2 fingers = RIGHT
            if direction != "LEFT":
                next_direction = "RIGHT"
        elif extended_count == 3:  # 3 fingers = DOWN
            if direction != "UP":
                next_direction = "DOWN"
        elif extended_count == 4:  # 4 fingers = LEFT
            if direction != "RIGHT":
                next_direction = "LEFT"
        elif extended_count == 5:  # 5 fingers = UP
            if direction != "DOWN":
                next_direction = "UP"

    # --- MOVE SNAKE (with delay for slower movement) ---
    if not paused and current_time - last_move_time > movement_delay:
        # Update direction from buffer
        direction = next_direction
        
        # Calculate new head position
        new_head = list(snake[0])
        if direction == "RIGHT":
            new_head[0] += SNAKE_SIZE
        elif direction == "LEFT":
            new_head[0] -= SNAKE_SIZE
        elif direction == "UP":
            new_head[1] -= SNAKE_SIZE
        elif direction == "DOWN":
            new_head[1] += SNAKE_SIZE

        snake.insert(0, new_head)
        last_move_time = current_time

        # FOOD & SCORE
        if new_head[0] == food[0] and new_head[1] == food[1]:
            score += 10
            high_score = max(high_score, score)
            food = [random.randrange(0, WIDTH, SNAKE_SIZE),
                    random.randrange(0, HEIGHT, SNAKE_SIZE)]
            print(f"🍎 Food eaten! Score: {score}")
        else:
            snake.pop()

        # COLLISIONS
        if (new_head[0] < 0 or new_head[0] >= WIDTH or
            new_head[1] < 0 or new_head[1] >= HEIGHT or
            new_head in snake[1:]):
            snake = [[200, 200], [180, 200], [160, 200]]
            direction = "RIGHT"
            next_direction = "RIGHT"
            score = 0
            print("💥 Game Over! Score reset")

    # --- DRAWING ---
    # Draw food with glow effect
    food_center = (food[0] + SNAKE_SIZE // 2, food[1] + SNAKE_SIZE // 2)
    pygame.draw.circle(screen, (255, 100, 100), food_center, SNAKE_SIZE // 2 + 4)
    pygame.draw.circle(screen, (255, 200, 200), food_center, SNAKE_SIZE // 2 + 2)
    pygame.draw.circle(screen, (255, 255, 255), food_center, SNAKE_SIZE // 2)
    
    # Draw snake with gradient effect
    for i, seg in enumerate(snake):
        center_x, center_y = seg[0] + SNAKE_SIZE // 2, seg[1] + SNAKE_SIZE // 2
        if i == 0:  # Head
            pulse = abs(int(5 * (current_time * 2) % 10 - 5)) / 5
            radius = int(SNAKE_SIZE // 2 + pulse * 2)
            pygame.draw.circle(screen, (200, 0, 255), (center_x, center_y), radius)
            pygame.draw.circle(screen, (255, 100, 255), (center_x, center_y), radius - 2)
        else:  # Body
            alpha = 1.0 - (i / len(snake)) * 0.5
            color = (int(150 * alpha), 0, int(200 * alpha))
            pygame.draw.circle(screen, color, (center_x, center_y), SNAKE_SIZE // 2)
            pygame.draw.circle(screen, (180, 50, 220), (center_x, center_y), SNAKE_SIZE // 2 - 2)

    # --- TEXT DISPLAY ---
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    high_text = font.render(f"High Score: {high_score}", True, (200, 200, 200))
    
    # Color-coded gesture display
    if extended_count <= 1:
        gesture_color = (255, 100, 100)  # Red for fist
    elif extended_count == 2:
        gesture_color = (100, 255, 100)  # Green for right
    elif extended_count == 3:
        gesture_color = (100, 100, 255)  # Blue for down
    elif extended_count == 4:
        gesture_color = (255, 255, 100)  # Yellow for left
    elif extended_count == 5:
        gesture_color = (255, 100, 255)  # Magenta for up
    else:
        gesture_color = (150, 150, 150)
    
    gesture_text = font.render(f"Gesture: {current_gesture} ({extended_count} fingers)", True, gesture_color)
    direction_text = font.render(f"Direction: {direction}", True, (100, 255, 100))
    speed_text = small_font.render(f"Speed: Slow", True, (200, 200, 200))
    
    # Pause indicator with overlay
    if paused:
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        pause_text = font.render("⏸️ PAUSED - Make fist to resume", True, (255, 100, 100))
        pause_rect = pause_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(pause_text, pause_rect)
        
        reset_hint = small_font.render("Open hand (5 fingers) to reset", True, (200, 200, 200))
        reset_rect = reset_hint.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))
        screen.blit(reset_hint, reset_rect)
    
    # Controls help
    y_offset = HEIGHT - 190
    controls_title = small_font.render("=== GESTURE CONTROLS ===", True, (255, 255, 0))
    screen.blit(controls_title, (10, y_offset))
    y_offset += 25
    
    controls = [
        "🤜 Fist (0-1 fingers) = Pause/Resume",
        "✌️ 2 Fingers = Move RIGHT",
        "🤟 3 Fingers = Move DOWN", 
        "🖖 4 Fingers = Move LEFT",
        "🖐️ Open Hand (5 fingers) = Move UP",
        "🔄 Open Hand (when paused) = Reset"
    ]
    
    for line in controls:
        control_text = small_font.render(line, True, (200, 200, 200))
        screen.blit(control_text, (20, y_offset))
        y_offset += 22
    
    screen.blit(score_text, (10, 10))
    screen.blit(high_text, (10, 40))
    screen.blit(gesture_text, (10, 70))
    screen.blit(direction_text, (10, 110))
    screen.blit(speed_text, (10, 140))

    # Show camera feed
    cv2.putText(frame, "Press 'Q' to quit, 'R' to reset", (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Hand Control Snake", frame)
    pygame.display.flip()

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        running = False
    elif key == ord("r"):
        snake = [[200, 200], [180, 200], [160, 200]]
        direction = "RIGHT"
        next_direction = "RIGHT"
        score = 0
        paused = False
        print("🔄 Manual reset")

    clock.tick(FPS)

# Cleanup
print("\n👋 Shutting down...")
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("Game closed. Thanks for playing!")