package com.posturewatch;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ScreenDistanceMonitor {

    // ── Detection constants ──────────────────────────────────────────────────
    private static final double DISTANCE_ALERT_RATIO = 1.20;
    private static final double HEAD_ANGLE_THRESHOLD = 0.09;
    private static final int    CALIBRATION_FRAMES   = 30;
    private static final int    CHECK_FRAMES         = 20;
    private static final int    CONSECUTIVE_REQUIRED = 8;
    private static final int    PW = 320, PH = 240;

    private static final String[] DISTANCE_TIPS = {
        "Move your monitor back to about arm's length — roughly 25 inches away.",
        "You're too close. Push the monitor back to about arm's length (~25 in).",
        "Scoot back until the screen is roughly arm's length away (~25 in)."
    };
    private static final String[] ANGLE_TIPS = {
        "Raise your monitor so the top of the screen is at eye level.",
        "Head tilting forward? Raise the monitor — top of screen at eye level cuts neck load.",
        "Lift the monitor until the top edge is level with your eyes."
    };

    private static volatile int distanceTipIdx = 0;
    private static volatile int angleTipIdx    = 0;

    private static final String[] FREQ_LABELS  = {"15s", "30s", "1m", "2m", "5m"};
    private static final int[]    FREQ_SECONDS = { 15,    30,   60,  120,  300 };
    private static volatile int   checkInterval = FREQ_SECONDS[1];

    // ── Theme ────────────────────────────────────────────────────────────────
    private static final Color[] ACCENT_PRESETS = {
        new Color(0,   210, 255),  // Cyan
        new Color(140,  80, 255),  // Purple
        new Color(0,   230, 130),  // Green
        new Color(255, 140,   0),  // Orange
        new Color(255,  60, 130),  // Pink
    };
    private static int     accentIdx = 0;
    private static boolean darkMode  = true;

    // Mutable design tokens — updated by applyTheme()
    private static Color BG         = new Color(9,   9,   13);
    private static Color BG2        = new Color(15,  15,  22);
    private static Color ACCENT     = ACCENT_PRESETS[0];
    private static Color ACCENT_DIM = new Color(0,   210, 255, 35);
    private static Color TEXT_HI    = new Color(230, 235, 255);
    private static Color TEXT_DIM   = new Color(90,  100, 130);
    private static Color TOGGLE_LBL = new Color(170, 175, 200);
    private static final Color GOOD = new Color(0,   230, 130);
    private static final Color BAD  = new Color(255,  55,  75);

    // ── App state ────────────────────────────────────────────────────────────
    private static volatile boolean monitoring     = false;
    private static volatile boolean showCamera     = false;
    private static volatile Double  baseline       = null;
    private static volatile Double  baselineYRatio = null;
    private static volatile long    nextCheckAt    = 0;
    private static volatile boolean ignoreFocusLoss = false;

    // ── UI references ────────────────────────────────────────────────────────
    private static JWindow      popup;
    private static JPanel       popupRoot;
    private static JLabel       previewLabel;
    private static JLabel       statusLabel;
    private static JLabel       countdownLabel;
    private static ToggleSwitch monitorToggle;
    private static ToggleSwitch cameraToggle;
    private static FreqSelector freqSelector;
    private static TrayIcon     trayIcon;
    // Theme-sensitive labels (updated by applyTheme)
    private static JLabel headerDotLabel;
    private static JLabel headerTitleLabel;
    private static JLabel checkEveryLabel;
    private static JLabel monitoringLabel;
    private static JLabel cameraLabel;
    private static JLabel appearanceLabel;
    private static JLabel accentColorLabel;
    private static JLabel lightModeLabel;
    // Tab buttons
    private static JButton tabMonitorBtn;
    private static JButton tabSettingsBtn;

    private static CascadeClassifier        faceDetector;
    private static ScheduledExecutorService checkScheduler;
    private static ScheduledExecutorService countdownScheduler;
    private static final CountDownLatch     shutdown = new CountDownLatch(1);

    // =========================================================================
    // Custom Components
    // =========================================================================

    static class ToggleSwitch extends JComponent {
        private boolean on;
        private float   anim;
        private Timer   timer;
        private final List<ActionListener> listeners = new ArrayList<>();

        ToggleSwitch(boolean init) {
            on = init; anim = init ? 1f : 0f;
            setPreferredSize(new Dimension(44, 26));
            setMaximumSize (new Dimension(44, 26));
            setMinimumSize (new Dimension(44, 26));
            setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            addMouseListener(new MouseAdapter() {
                @Override public void mousePressed(MouseEvent e) {
                    if (!isEnabled()) return;
                    on = !on;
                    animateTo(on ? 1f : 0f);
                    ActionEvent ae = new ActionEvent(ToggleSwitch.this, 0, on ? "on" : "off");
                    listeners.forEach(l -> l.actionPerformed(ae));
                }
            });
        }

        void setOn(boolean v) { if (on != v) { on = v; animateTo(v ? 1f : 0f); } }
        boolean isOn() { return on; }
        void addActionListener(ActionListener l) { listeners.add(l); }

        private void animateTo(float target) {
            if (timer != null) timer.stop();
            timer = new Timer(14, null);
            timer.addActionListener(e -> {
                float d = target - anim;
                if (Math.abs(d) < 0.025f) { anim = target; timer.stop(); }
                else anim += d * 0.28f;
                repaint();
            });
            timer.start();
        }

        @Override protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            int w = getWidth(), h = getHeight();
            float a = isEnabled() ? 1f : 0.3f;
            Color ac = ACCENT;
            int offR = darkMode ? 30 : 180, offG = darkMode ? 30 : 180, offB = darkMode ? 40 : 190;
            int r  = lerp(offR, ac.getRed(),   anim);
            int gr = lerp(offG, ac.getGreen(), anim);
            int b  = lerp(offB, ac.getBlue(),  anim);
            g2.setColor(new Color(r, gr, b, (int)(255 * a)));
            g2.fillRoundRect(0, 0, w, h, h, h);
            if (anim > 0.1f) {
                g2.setColor(new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), (int)(40 * anim * a)));
                g2.setStroke(new BasicStroke(3));
                g2.drawRoundRect(-1, -1, w+2, h+2, h+2, h+2);
            }
            int td = h - 4, tx = 2 + (int)(anim * (w - td - 4));
            g2.setColor(new Color(0, 0, 0, 60));
            g2.fillOval(tx + 1, 3, td, td);
            g2.setColor(new Color(255, 255, 255, (int)(255 * a)));
            g2.fillOval(tx, 2, td, td);
            g2.dispose();
        }

        private static int lerp(int a, int b, float t) {
            return Math.max(0, Math.min(255, (int)(a + t * (b - a))));
        }
    }

    static class FreqSelector extends JPanel {
        private int selected;
        private final JButton[] btns = new JButton[FREQ_LABELS.length];
        private ActionListener listener;

        FreqSelector(int defaultIdx) {
            selected = defaultIdx;
            setOpaque(false);
            setLayout(new FlowLayout(FlowLayout.LEFT, 5, 0));
            for (int i = 0; i < FREQ_LABELS.length; i++) {
                final int idx = i;
                btns[i] = pill(FREQ_LABELS[i]);
                btns[i].addActionListener(e -> select(idx));
                add(btns[i]);
            }
            refresh();
        }

        void select(int idx) {
            selected = idx;
            refresh();
            if (listener != null)
                listener.actionPerformed(new ActionEvent(this, 0, String.valueOf(FREQ_SECONDS[idx])));
        }

        int getSelected() { return selected; }
        void addActionListener(ActionListener l) { listener = l; }

        private void refresh() {
            for (int i = 0; i < btns.length; i++) {
                btns[i].putClientProperty("active", i == selected);
                btns[i].repaint();
            }
        }

        static JButton pill(String text) {
            JButton b = new JButton(text) {
                @Override protected void paintComponent(Graphics g) {
                    boolean active = Boolean.TRUE.equals(getClientProperty("active"));
                    Graphics2D g2 = (Graphics2D) g.create();
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    int w = getWidth(), h = getHeight();
                    Color ac = ACCENT;
                    if (active) {
                        g2.setColor(new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), 35));
                        g2.fillRoundRect(-2, -2, w+4, h+4, h+4, h+4);
                        g2.setColor(darkMode ? new Color(0, 60, 80) : new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), 50));
                        g2.fillRoundRect(0, 0, w, h, h, h);
                        g2.setColor(ac);
                        g2.setStroke(new BasicStroke(1.2f));
                        g2.drawRoundRect(0, 0, w-1, h-1, h-1, h-1);
                        g2.setFont(getFont().deriveFont(Font.BOLD));
                        g2.setColor(ac);
                    } else {
                        g2.setColor(darkMode ? new Color(25, 25, 36) : new Color(218, 220, 228));
                        g2.fillRoundRect(0, 0, w, h, h, h);
                        g2.setColor(darkMode ? new Color(60, 65, 85) : new Color(180, 183, 197));
                        g2.setStroke(new BasicStroke(1.0f));
                        g2.drawRoundRect(0, 0, w-1, h-1, h-1, h-1);
                        g2.setFont(getFont());
                        g2.setColor(TEXT_DIM);
                    }
                    FontMetrics fm = g2.getFontMetrics();
                    int tx = (w - fm.stringWidth(getText())) / 2;
                    int ty = (h - fm.getHeight()) / 2 + fm.getAscent();
                    g2.drawString(getText(), tx, ty);
                    g2.dispose();
                }
            };
            b.setOpaque(false); b.setContentAreaFilled(false);
            b.setBorderPainted(false); b.setFocusPainted(false);
            b.setFont(b.getFont().deriveFont(11f));
            b.setPreferredSize(new Dimension(46, 24));
            b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            return b;
        }
    }

    static class Divider extends JComponent {
        Divider() {
            setMaximumSize(new Dimension(Integer.MAX_VALUE, 1));
            setPreferredSize(new Dimension(0, 1));
        }
        @Override protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            Color ac = ACCENT;
            g2.setPaint(new GradientPaint(
                    0, 0, new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), 0),
                    getWidth() / 2f, 0, new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), 45)));
            g2.fillRect(0, 0, getWidth(), 1);
            g2.dispose();
        }
    }

    // =========================================================================
    // Entry point
    // =========================================================================

    public static void main(String[] args) throws Exception {
        if (!SystemTray.isSupported()) { System.err.println("System tray not supported!"); return; }
        SwingUtilities.invokeAndWait(() -> {
            try { buildPopup(); buildTray(); }
            catch (Exception e) { e.printStackTrace(); }
        });
        shutdown.await();
        System.exit(0);
    }

    // =========================================================================
    // UI construction
    // =========================================================================

    private static void buildPopup() {
        popup = new JWindow();
        popup.setBackground(BG);
        popup.setAlwaysOnTop(true);

        popupRoot = new JPanel() {
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(BG);
                g2.fillRect(0, 0, getWidth(), getHeight());
                Color ac = ACCENT;
                GradientPaint shine = new GradientPaint(
                        0, 0, new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), darkMode ? 22 : 12),
                        0, 60, new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), 0));
                g2.setPaint(shine);
                g2.fillRect(0, 0, getWidth(), 60);
                g2.setColor(new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), 18));
                g2.fillRect(getWidth() - 1, 0, 1, getHeight());
                g2.dispose();
            }
        };
        popupRoot.setOpaque(true);
        popupRoot.setBackground(BG);
        popupRoot.setLayout(new BoxLayout(popupRoot, BoxLayout.Y_AXIS));
        popupRoot.setBorder(new EmptyBorder(14, 16, 14, 16));

        // ── Header ───────────────────────────────────────────────────────────
        JPanel header = row();
        headerDotLabel = new JLabel("◈");
        headerDotLabel.setForeground(ACCENT);
        headerDotLabel.setFont(headerDotLabel.getFont().deriveFont(Font.BOLD, 13f));
        headerTitleLabel = label("POSTUREGUARD", TEXT_HI, Font.BOLD, 13f);
        headerTitleLabel.setBorder(new EmptyBorder(0, 6, 0, 0));
        header.add(headerDotLabel);
        header.add(headerTitleLabel);
        header.add(Box.createHorizontalGlue());
        popupRoot.add(header);
        popupRoot.add(vgap(8));
        popupRoot.add(new Divider());
        popupRoot.add(vgap(8));

        // ── Tab bar ───────────────────────────────────────────────────────────
        JPanel tabBar = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
        tabBar.setOpaque(false);
        tabBar.setAlignmentX(Component.LEFT_ALIGNMENT);
        tabBar.setMaximumSize(new Dimension(Integer.MAX_VALUE, 30));
        tabMonitorBtn  = tabButton("MONITOR",  true);
        tabSettingsBtn = tabButton("SETTINGS", false);
        tabBar.add(tabMonitorBtn);
        tabBar.add(tabSettingsBtn);
        popupRoot.add(tabBar);
        popupRoot.add(vgap(10));

        // ── Swappable content area ────────────────────────────────────────────
        JPanel contentArea = new JPanel(new BorderLayout());
        contentArea.setOpaque(false);
        contentArea.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel monitorPanel  = buildMonitorPanel();
        JPanel settingsPanel = buildSettingsPanel();
        contentArea.add(monitorPanel, BorderLayout.CENTER);
        popupRoot.add(contentArea);

        tabMonitorBtn.addActionListener(e -> {
            contentArea.removeAll();
            contentArea.add(monitorPanel, BorderLayout.CENTER);
            tabMonitorBtn.putClientProperty("active", true);
            tabSettingsBtn.putClientProperty("active", false);
            tabMonitorBtn.repaint(); tabSettingsBtn.repaint();
            repackPopup();
        });
        tabSettingsBtn.addActionListener(e -> {
            contentArea.removeAll();
            contentArea.add(settingsPanel, BorderLayout.CENTER);
            tabMonitorBtn.putClientProperty("active", false);
            tabSettingsBtn.putClientProperty("active", true);
            tabMonitorBtn.repaint(); tabSettingsBtn.repaint();
            repackPopup();
        });

        // Make popup draggable
        MouseAdapter drag = new MouseAdapter() {
            Point origin;
            @Override public void mousePressed(MouseEvent e)  { origin = e.getPoint(); }
            @Override public void mouseDragged(MouseEvent e)  {
                if (origin == null) return;
                Point loc = popup.getLocationOnScreen();
                popup.setLocation(loc.x + e.getX() - origin.x, loc.y + e.getY() - origin.y);
            }
        };
        popupRoot.addMouseListener(drag);
        popupRoot.addMouseMotionListener(drag);

        popup.setContentPane(popupRoot);
        popup.pack();
        popup.setShape(new RoundRectangle2D.Float(0, 0, popup.getWidth(), popup.getHeight(), 18, 18));

        popup.addWindowFocusListener(new WindowAdapter() {
            @Override public void windowLostFocus(WindowEvent e) {
                if (!ignoreFocusLoss) popup.setVisible(false);
            }
        });
    }

    private static JPanel buildMonitorPanel() {
        JPanel p = new JPanel();
        p.setOpaque(false);
        p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));

        // Camera preview
        previewLabel = new JLabel("CAMERA OFF") {
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(BG2);
                g2.fillRoundRect(0, 0, getWidth(), getHeight(), 10, 10);
                int m = 8, len = 18;
                Color ac = ACCENT;
                g2.setColor(new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), 90));
                g2.setStroke(new BasicStroke(1.5f));
                g2.drawLine(m, m, m+len, m);               g2.drawLine(m, m, m, m+len);
                g2.drawLine(getWidth()-m-len, m, getWidth()-m, m);
                g2.drawLine(getWidth()-m, m, getWidth()-m, m+len);
                g2.drawLine(m, getHeight()-m, m+len, getHeight()-m);
                g2.drawLine(m, getHeight()-m-len, m, getHeight()-m);
                g2.drawLine(getWidth()-m-len, getHeight()-m, getWidth()-m, getHeight()-m);
                g2.drawLine(getWidth()-m, getHeight()-m-len, getWidth()-m, getHeight()-m);
                super.paintComponent(g);
                g2.dispose();
            }
        };
        previewLabel.setPreferredSize(new Dimension(PW, PH));
        previewLabel.setMinimumSize(new Dimension(PW, PH));
        previewLabel.setMaximumSize(new Dimension(PW, PH));
        previewLabel.setOpaque(false);
        previewLabel.setForeground(new Color(40, 50, 70));
        previewLabel.setFont(new Font("Monospaced", Font.BOLD, 11));
        previewLabel.setHorizontalAlignment(SwingConstants.CENTER);
        previewLabel.setVerticalAlignment(SwingConstants.CENTER);
        p.add(previewLabel);
        p.add(vgap(10));

        // Status + countdown
        JPanel statusRow = row();
        statusLabel    = label("OFFLINE", TEXT_DIM, Font.BOLD, 11f);
        countdownLabel = label("", TEXT_DIM, Font.PLAIN, 11f);
        statusRow.add(statusLabel);
        statusRow.add(Box.createHorizontalGlue());
        statusRow.add(countdownLabel);
        p.add(statusRow);
        p.add(vgap(10));
        p.add(new Divider());
        p.add(vgap(10));

        // Frequency selector
        checkEveryLabel = label("CHECK EVERY", TEXT_DIM, Font.BOLD, 10f);
        p.add(checkEveryLabel);
        p.add(vgap(6));
        freqSelector = new FreqSelector(1);
        freqSelector.setAlignmentX(Component.LEFT_ALIGNMENT);
        freqSelector.addActionListener(e -> {
            checkInterval = FREQ_SECONDS[freqSelector.getSelected()];
            if (monitoring) restartCheckScheduler();
        });
        p.add(freqSelector);
        p.add(vgap(10));
        p.add(new Divider());
        p.add(vgap(10));

        // Monitoring toggle
        monitorToggle = new ToggleSwitch(false);
        monitorToggle.addActionListener(e -> onMonitorToggle());
        monitoringLabel = label("MONITORING", TOGGLE_LBL, Font.BOLD, 11f);
        JPanel monRow = row();
        monRow.add(monitoringLabel);
        monRow.add(Box.createHorizontalGlue());
        monRow.add(monitorToggle);
        p.add(monRow);
        p.add(vgap(8));

        // Camera toggle
        cameraToggle = new ToggleSwitch(false);
        cameraToggle.setEnabled(false);
        cameraToggle.addActionListener(e -> {
            showCamera = cameraToggle.isOn();
            if (!showCamera) clearPreview();
        });
        cameraLabel = label("CAMERA VIEW", TOGGLE_LBL, Font.BOLD, 11f);
        JPanel camRow = row();
        camRow.add(cameraLabel);
        camRow.add(Box.createHorizontalGlue());
        camRow.add(cameraToggle);
        p.add(camRow);
        p.add(vgap(10));
        p.add(new Divider());
        p.add(vgap(10));

        // Quit
        JButton quit = buildQuitButton();
        quit.addActionListener(e -> doQuit());
        p.add(quit);
        return p;
    }

    private static JPanel buildSettingsPanel() {
        JPanel p = new JPanel();
        p.setOpaque(false);
        p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));

        // Appearance section
        appearanceLabel = label("APPEARANCE", TEXT_DIM, Font.BOLD, 10f);
        p.add(appearanceLabel);
        p.add(vgap(8));

        ToggleSwitch themeToggle = new ToggleSwitch(!darkMode); // on = light mode
        themeToggle.addActionListener(e -> {
            darkMode = !themeToggle.isOn();
            applyTheme();
        });
        lightModeLabel = label("LIGHT MODE", TOGGLE_LBL, Font.BOLD, 11f);
        JPanel themeRow = row();
        themeRow.add(lightModeLabel);
        themeRow.add(Box.createHorizontalGlue());
        themeRow.add(themeToggle);
        p.add(themeRow);
        p.add(vgap(14));
        p.add(new Divider());
        p.add(vgap(12));

        // Accent color section
        accentColorLabel = label("ACCENT COLOR", TEXT_DIM, Font.BOLD, 10f);
        p.add(accentColorLabel);
        p.add(vgap(8));

        JPanel swatches = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        swatches.setOpaque(false);
        swatches.setAlignmentX(Component.LEFT_ALIGNMENT);
        String[] names = {"Cyan", "Purple", "Green", "Orange", "Pink"};
        for (int i = 0; i < ACCENT_PRESETS.length; i++) {
            final int idx = i;
            final Color c = ACCENT_PRESETS[i];
            JComponent swatch = new JComponent() {
                @Override protected void paintComponent(Graphics g) {
                    Graphics2D g2 = (Graphics2D) g.create();
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    if (accentIdx == idx) {
                        // selection ring
                        g2.setColor(c);
                        g2.setStroke(new BasicStroke(2f));
                        g2.drawOval(1, 1, getWidth()-3, getHeight()-3);
                    }
                    g2.setColor(c);
                    g2.fillOval(5, 5, getWidth()-10, getHeight()-10);
                    g2.dispose();
                }
            };
            swatch.setPreferredSize(new Dimension(32, 32));
            swatch.setToolTipText(names[i]);
            swatch.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            swatch.addMouseListener(new MouseAdapter() {
                @Override public void mousePressed(MouseEvent e) {
                    accentIdx = idx;
                    applyTheme();
                    swatches.repaint();
                }
            });
            swatches.add(swatch);
        }
        p.add(swatches);
        p.add(vgap(14));
        p.add(new Divider());
        p.add(vgap(10));

        // Quit from settings tab too
        JButton quit = buildQuitButton();
        quit.addActionListener(e -> doQuit());
        p.add(quit);
        return p;
    }

    private static void applyTheme() {
        ACCENT     = ACCENT_PRESETS[accentIdx];
        ACCENT_DIM = new Color(ACCENT.getRed(), ACCENT.getGreen(), ACCENT.getBlue(), 35);
        if (darkMode) {
            BG         = new Color(9,   9,   13);
            BG2        = new Color(15,  15,  22);
            TEXT_HI    = new Color(230, 235, 255);
            TEXT_DIM   = new Color(90,  100, 130);
            TOGGLE_LBL = new Color(170, 175, 200);
        } else {
            BG         = new Color(242, 242, 247);
            BG2        = new Color(255, 255, 255);
            TEXT_HI    = new Color(20,  20,  30);
            TEXT_DIM   = new Color(100, 105, 120);
            TOGGLE_LBL = new Color(50,  55,  75);
        }
        SwingUtilities.invokeLater(() -> {
            if (popup == null) return;
            popup.setBackground(BG);
            popupRoot.setBackground(BG);
            if (headerDotLabel   != null) headerDotLabel.setForeground(ACCENT);
            if (headerTitleLabel != null) headerTitleLabel.setForeground(TEXT_HI);
            if (countdownLabel   != null) countdownLabel.setForeground(TEXT_DIM);
            if (checkEveryLabel  != null) checkEveryLabel.setForeground(TEXT_DIM);
            if (monitoringLabel  != null) monitoringLabel.setForeground(TOGGLE_LBL);
            if (cameraLabel      != null) cameraLabel.setForeground(TOGGLE_LBL);
            if (appearanceLabel  != null) appearanceLabel.setForeground(TEXT_DIM);
            if (accentColorLabel != null) accentColorLabel.setForeground(TEXT_DIM);
            if (lightModeLabel   != null) lightModeLabel.setForeground(TOGGLE_LBL);
            if (statusLabel != null && !monitoring) statusLabel.setForeground(TEXT_DIM);
            popup.repaint();
        });
    }

    private static JButton tabButton(String text, boolean active) {
        JButton b = new JButton(text) {
            @Override protected void paintComponent(Graphics g) {
                boolean a = Boolean.TRUE.equals(getClientProperty("active"));
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                int w = getWidth(), h = getHeight();
                Color ac = ACCENT;
                if (a) {
                    g2.setColor(new Color(ac.getRed(), ac.getGreen(), ac.getBlue(), 30));
                    g2.fillRoundRect(0, 0, w, h, 8, 8);
                    g2.setColor(ac);
                    g2.setStroke(new BasicStroke(1.2f));
                    g2.drawRoundRect(0, 0, w-1, h-1, 8, 8);
                    g2.setFont(getFont().deriveFont(Font.BOLD, 10f));
                    g2.setColor(ac);
                } else {
                    g2.setColor(darkMode ? new Color(25, 25, 36) : new Color(215, 217, 225));
                    g2.fillRoundRect(0, 0, w, h, 8, 8);
                    g2.setFont(getFont().deriveFont(10f));
                    g2.setColor(TEXT_DIM);
                }
                FontMetrics fm = g2.getFontMetrics();
                g2.drawString(getText(),
                        (w - fm.stringWidth(getText())) / 2,
                        (h - fm.getHeight()) / 2 + fm.getAscent());
                g2.dispose();
            }
        };
        b.putClientProperty("active", active);
        b.setOpaque(false); b.setContentAreaFilled(false);
        b.setBorderPainted(false); b.setFocusPainted(false);
        b.setPreferredSize(new Dimension(90, 26));
        b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        return b;
    }

    private static JButton buildQuitButton() {
        JButton quit = new JButton("QUIT") {
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                boolean pressed = getModel().isPressed();
                g2.setColor(pressed ? new Color(180, 30, 50) : new Color(35, 15, 18));
                g2.fillRoundRect(0, 0, getWidth(), getHeight(), 8, 8);
                g2.setColor(pressed ? new Color(255, 80, 100) : new Color(120, 35, 45));
                g2.setStroke(new BasicStroke(1.1f));
                g2.drawRoundRect(0, 0, getWidth()-1, getHeight()-1, 8, 8);
                g2.setFont(getFont().deriveFont(Font.BOLD, 11f));
                g2.setColor(pressed ? Color.WHITE : new Color(200, 80, 100));
                FontMetrics fm = g2.getFontMetrics();
                g2.drawString(getText(),
                        (getWidth() - fm.stringWidth(getText())) / 2,
                        (getHeight() - fm.getHeight()) / 2 + fm.getAscent());
                g2.dispose();
            }
        };
        quit.setOpaque(false); quit.setContentAreaFilled(false);
        quit.setBorderPainted(false); quit.setFocusPainted(false);
        quit.setPreferredSize(new Dimension(100, 30));
        quit.setMaximumSize(new Dimension(Integer.MAX_VALUE, 30));
        quit.setAlignmentX(Component.CENTER_ALIGNMENT);
        quit.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        return quit;
    }

    private static void doQuit() {
        popup.setVisible(false);
        stopMonitoring();
        try { SystemTray.getSystemTray().remove(trayIcon); } catch (Exception ignored) {}
        shutdown.countDown();
    }

    private static void repackPopup() {
        SwingUtilities.invokeLater(() -> {
            Point loc = popup.getLocation();
            popup.pack();
            popup.setLocation(loc);
            popup.setShape(new RoundRectangle2D.Float(0, 0, popup.getWidth(), popup.getHeight(), 18, 18));
        });
    }

    private static void buildTray() throws AWTException {
        SystemTray tray = SystemTray.getSystemTray();
        trayIcon = new TrayIcon(buildIcon(tray.getTrayIconSize()), "PostureGuard");
        trayIcon.setImageAutoSize(true);
        trayIcon.addMouseListener(new MouseAdapter() {
            @Override public void mousePressed(MouseEvent e) {
                if (popup.isVisible()) {
                    popup.setVisible(false);
                } else {
                    SwingUtilities.invokeLater(() -> {
                        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
                        int x = Math.max(0, Math.min(e.getX() - popup.getWidth() / 2,
                                                      screen.width - popup.getWidth()));
                        popup.setLocation(x, e.getY() + 5);
                        popup.setVisible(true);
                        popup.toFront();
                        popup.requestFocus();
                    });
                }
            }
        });
        tray.add(trayIcon);
    }

    // =========================================================================
    // Monitoring control
    // =========================================================================

    private static void onMonitorToggle() {
        monitoring = monitorToggle.isOn();
        if (monitoring) {
            String cascadePath = new File("src/main/resources/haarcascade_frontalface_default.xml")
                    .getAbsolutePath();
            faceDetector = new CascadeClassifier(cascadePath);
            if (faceDetector.empty()) {
                osNotify("Error: face detector file not found.");
                monitorToggle.setOn(false); monitoring = false; return;
            }
            baseline = null; baselineYRatio = null; nextCheckAt = 0;
            cameraToggle.setEnabled(true);
            startSchedulers();
            setStatus("STARTING...", TEXT_DIM);
        } else {
            cameraToggle.setEnabled(false);
            cameraToggle.setOn(false);
            showCamera = false;
            stopMonitoring();
            clearPreview();
            setStatus("OFFLINE", TEXT_DIM);
            SwingUtilities.invokeLater(() -> countdownLabel.setText(""));
        }
    }

    private static void startSchedulers() {
        checkScheduler = daemonScheduler("posture-check");
        checkScheduler.scheduleWithFixedDelay(
                ScreenDistanceMonitor::runCheck, 0, checkInterval, TimeUnit.SECONDS);
        countdownScheduler = daemonScheduler("countdown");
        countdownScheduler.scheduleAtFixedRate(
                ScreenDistanceMonitor::tickCountdown, 1, 1, TimeUnit.SECONDS);
    }

    private static void restartCheckScheduler() {
        if (checkScheduler != null) checkScheduler.shutdownNow();
        checkScheduler = daemonScheduler("posture-check");
        nextCheckAt = System.currentTimeMillis() + checkInterval * 1000L;
        checkScheduler.scheduleWithFixedDelay(
                ScreenDistanceMonitor::runCheck, checkInterval, checkInterval, TimeUnit.SECONDS);
    }

    private static void stopMonitoring() {
        monitoring = false; nextCheckAt = 0;
        if (checkScheduler     != null) { checkScheduler.shutdownNow();     checkScheduler     = null; }
        if (countdownScheduler != null) { countdownScheduler.shutdownNow(); countdownScheduler = null; }
    }

    private static ScheduledExecutorService daemonScheduler(String name) {
        return Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, name); t.setDaemon(true); return t;
        });
    }

    // =========================================================================
    // Countdown
    // =========================================================================

    private static void tickCountdown() {
        if (!monitoring || nextCheckAt == 0) return;
        long secs = (nextCheckAt - System.currentTimeMillis()) / 1000;
        if (secs > 0) {
            String txt = secs >= 60
                    ? secs / 60 + "m " + String.format("%02d", secs % 60) + "s"
                    : String.format("%02d", secs) + "s";
            SwingUtilities.invokeLater(() -> countdownLabel.setText(txt));
        }
    }

    // =========================================================================
    // Check logic
    // =========================================================================

    private static void runCheck() {
        if (!monitoring) return;
        setStatus("SCANNING...", ACCENT);
        SwingUtilities.invokeLater(() -> countdownLabel.setText(""));
        try {
            OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
            grabber.start();
            try {
                doCheck(grabber);
            } finally {
                if (showCamera) { Thread.sleep(1200); clearPreview(); }
                grabber.stop();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            if (monitoring) e.printStackTrace();
        }
        if (monitoring) nextCheckAt = System.currentTimeMillis() + checkInterval * 1000L;
    }

    private static void doCheck(OpenCVFrameGrabber grabber) throws Exception {
        OpenCVFrameConverter.ToMat matConv = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter java2D = new Java2DFrameConverter();

        boolean calibrating = (baseline == null);
        int totalFrames = calibrating ? CALIBRATION_FRAMES + CHECK_FRAMES : CHECK_FRAMES;
        int frameH = grabber.getImageHeight();

        double calWidthSum = 0, calYSum = 0;
        int calCount = 0;
        int consecutiveClose = 0, consecutiveHeadDown = 0;
        boolean tooClose = false, headAngle = false;

        for (int i = 0; i < totalFrames && monitoring; i++) {
            Frame frame = grabber.grab();
            if (frame == null) continue;
            Mat mat = matConv.convert(frame);
            if (mat == null) continue;

            Mat gray = new Mat();
            cvtColor(mat, gray, COLOR_BGR2GRAY);
            equalizeHist(gray, gray);

            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(gray, faces);

            Scalar boxColor = null;
            if (faces.size() > 0) {
                Rect largest = faces.get(0);
                for (long fi = 1; fi < faces.size(); fi++) {
                    if (faces.get(fi).width() > largest.width()) largest = faces.get(fi);
                }
                int width = largest.width();
                double yRatio = frameH > 0
                        ? (largest.y() + largest.height() / 2.0) / frameH
                        : 0.5;

                if (calibrating && baseline == null) {
                    calWidthSum += width; calYSum += yRatio; calCount++;
                    boxColor = new Scalar(255, 200, 0, 0); // amber: calibrating
                    if (calCount >= CALIBRATION_FRAMES) {
                        baseline       = calWidthSum / calCount;
                        baselineYRatio = calYSum     / calCount;
                        setStatus("CALIBRATED", ACCENT);
                    }
                } else if (baseline != null && baselineYRatio != null) {
                    boolean frameClose    = (width / baseline) >= DISTANCE_ALERT_RATIO;
                    // Positive delta → face dropped lower in frame → head angled forward/down
                    boolean frameHeadDown = (yRatio - baselineYRatio) > HEAD_ANGLE_THRESHOLD;
                    consecutiveClose    = frameClose    ? consecutiveClose + 1    : 0;
                    consecutiveHeadDown = frameHeadDown ? consecutiveHeadDown + 1 : 0;
                    if (consecutiveClose    >= CONSECUTIVE_REQUIRED) tooClose  = true;
                    if (consecutiveHeadDown >= CONSECUTIVE_REQUIRED) headAngle = true;
                    boxColor = tooClose  ? new Scalar(75,  55, 255, 0)
                             : headAngle ? new Scalar(0,  165, 255, 0)
                                        : new Scalar(0,  230, 130, 0);
                }
                if (showCamera && boxColor != null) rectangle(mat, largest, boxColor, 2, LINE_8, 0);
            }

            if (showCamera) {
                BufferedImage bi = java2D.convert(matConv.convert(mat));
                if (bi != null) {
                    Image scaled = bi.getScaledInstance(PW, PH, Image.SCALE_FAST);
                    SwingUtilities.invokeLater(() -> {
                        previewLabel.setIcon(new ImageIcon(scaled));
                        previewLabel.setText(null);
                    });
                }
            }
        }

        if (baseline == null) return; // only finished calibration, no result yet

        // Notify every check — no cooldown
        if (tooClose) {
            String tip = DISTANCE_TIPS[distanceTipIdx % DISTANCE_TIPS.length];
            distanceTipIdx++;
            setStatus("TOO CLOSE", BAD);
            osNotify(tip);
        } else if (headAngle) {
            String tip = ANGLE_TIPS[angleTipIdx % ANGLE_TIPS.length];
            angleTipIdx++;
            setStatus("HEAD FORWARD", BAD);
            osNotify(tip);
        } else {
            setStatus("ALL CLEAR", GOOD);
            osNotify("Good posture! Keep it up.");
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static void setStatus(String text, Color color) {
        SwingUtilities.invokeLater(() -> {
            statusLabel.setText(text);
            statusLabel.setForeground(color);
        });
    }

    private static void clearPreview() {
        SwingUtilities.invokeLater(() -> {
            previewLabel.setIcon(null);
            previewLabel.setText("CAMERA OFF");
        });
    }

    private static void osNotify(String msg) {
        ignoreFocusLoss = true;
        try {
            new ProcessBuilder("osascript", "-e",
                    "display notification \"" + msg + "\" with title \"PostureGuard\"").start();
        } catch (Exception ignored) {}
        new Thread(() -> {
            try { Thread.sleep(3000); } catch (Exception ignored) {}
            ignoreFocusLoss = false;
        }, "focus-reset").start();
    }

    private static JLabel label(String text, Color color, int style, float size) {
        JLabel l = new JLabel(text);
        l.setForeground(color);
        l.setFont(l.getFont().deriveFont(style, size));
        return l;
    }

    private static JPanel row() {
        JPanel p = new JPanel();
        p.setLayout(new BoxLayout(p, BoxLayout.X_AXIS));
        p.setOpaque(false);
        p.setAlignmentX(Component.LEFT_ALIGNMENT);
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, 28));
        return p;
    }

    private static Component vgap(int h) { return Box.createVerticalStrut(h); }

    private static Image buildIcon(Dimension size) {
        int s = Math.max(Math.max(size.width, size.height), 16) * 4;
        BufferedImage img = new BufferedImage(s, s, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        int hd = s * 3 / 10;
        g.fillOval(s/2 - hd/2, s/10, hd, hd);
        int sw = s/8, st = s/10 + hd + s/16, sh = s*4/10;
        g.fillRoundRect(s/2 - sw/2, st, sw, sh, s/8, s/8);
        g.setStroke(new BasicStroke(s/8f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g.drawLine(s/6, st + sh/6, 5*s/6, st + sh/6);
        g.dispose();
        int w = size.width > 0 ? size.width : 22, h = size.height > 0 ? size.height : 22;
        return img.getScaledInstance(w, h, Image.SCALE_SMOOTH);
    }
}
