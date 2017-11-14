package com.example.g.cardet;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;

/**
 * Created by ictlab on 2017-05-28.
 */

public class socketManager {
    private static String SERVER_IP = "114.70.21.231";
    private Socket sock = null;
    private DataInputStream server_dis = null;
    private DataOutputStream server_dos = null;

    public void connectServerSocket(int port) throws Exception {
        sock = new Socket(SERVER_IP, port);

        server_dis = new DataInputStream(sock.getInputStream());
        server_dos = new DataOutputStream(sock.getOutputStream());
    }

    public Socket getServerSocket() {
        return sock;
    }

    public DataInputStream getServer_dis() {
        return server_dis;
    }

    public DataOutputStream getServer_dos() {
        return server_dos;
    }
}
