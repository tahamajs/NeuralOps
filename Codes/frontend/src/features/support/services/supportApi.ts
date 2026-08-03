// src/features/support/services/supportApi.ts
import { request } from "../../../api/client";

export interface TicketMessage {
  id: string;
  ticket: string;
  role: "user" | "agent" | "system";
  content: string;
  metadata?: any;
  created_at: string;
}

export interface SupportTicket {
  id: string;
  user: number;
  username: string;
  title: string;
  description: string;
  code_snippet?: string;
  log_attachment?: string;
  tags: string[];
  status: "open" | "in_progress" | "resolved" | "closed";
  ai_summary?: string;
  ai_steps?: string[];
  web_sources?: any;
  messages: TicketMessage[];
  created_at: string;
  updated_at: string;
}

export interface Solution {
  id: string;
  ticket: string;
  ticket_title: string;
  content: string;
  upvotes: number;
  created_at: string;
}

export interface Feedback {
  id: string;
  ticket: string;
  rating: number;
  comment?: string;
}

export const supportAPI = {
  getTickets: async (): Promise<SupportTicket[]> => {
    return [
      {
        id: "TICK-1234",
        user: 1,
        username: "user",
        title: "Connection timeout on cluster A",
        description: "Getting timeout when submitting jobs.",
        tags: ["infrastructure", "networking"],
        status: "open",
        messages: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ];
  },

  createTicket: async (data: {
    title: string;
    description: string;
    code_snippet?: string;
    log_attachment?: string;
    tags?: string[];
  }): Promise<SupportTicket> => {
    return {
      id: `TICK-${Math.floor(Math.random() * 10000)}`,
      user: 1,
      username: "user",
      title: data.title,
      description: data.description,
      tags: data.tags || [],
      status: "open",
      messages: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
  },

  getTicketDetail: async (ticketId: string): Promise<SupportTicket> => {
    return {
      id: ticketId,
      user: 1,
      username: "user",
      title: "Connection timeout on cluster A",
      description: "Getting timeout when submitting jobs.",
      tags: ["infrastructure", "networking"],
      status: "open",
      messages: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
  },

  getTicketMessages: async (ticketId: string): Promise<TicketMessage[]> => {
    return [];
  },

  sendTicketMessage: async (
    ticketId: string,
    content: string,
  ): Promise<TicketMessage> => {
    return {
      id: "msg-123",
      ticket: ticketId,
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };
  },

  submitFeedback: async (
    ticketId: string,
    rating: number,
    comment?: string,
  ): Promise<any> => {
    return { success: true };
  },

  getSolutions: async (): Promise<Solution[]> => {
    return [
      {
        id: "SOL-1",
        ticket: "TICK-0001",
        ticket_title: "How to fix CUDA out of memory?",
        content: "Reduce batch size or use mixed precision training (fp16).",
        upvotes: 42,
        created_at: new Date().toISOString(),
      },
    ];
  },

  upvoteSolution: async (solutionId: string): Promise<any> => {
    return { success: true };
  },
};
