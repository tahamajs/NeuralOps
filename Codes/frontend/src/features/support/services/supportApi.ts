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
  getTickets: (): Promise<SupportTicket[]> => {
    return request.get("/support/tickets/");
  },
  
  createTicket: (data: {
    title: string;
    description: string;
    code_snippet?: string;
    log_attachment?: string;
    tags?: string[];
  }): Promise<SupportTicket> => {
    return request.post("/support/tickets/", data);
  },
  
  getTicketDetail: (ticketId: string): Promise<SupportTicket> => {
    return request.get(`/support/tickets/${ticketId}/`);
  },
  
  getTicketMessages: (ticketId: string): Promise<TicketMessage[]> => {
    return request.get(`/support/tickets/${ticketId}/messages/`);
  },
  
  sendTicketMessage: (ticketId: string, content: string): Promise<TicketMessage> => {
    return request.post(`/support/tickets/${ticketId}/messages/`, { content });
  },
  
  submitFeedback: (ticketId: string, rating: number, comment?: string): Promise<any> => {
    return request.post(`/support/tickets/${ticketId}/feedback/`, { rating, comment });
  },
  
  getSolutions: (): Promise<Solution[]> => {
    return request.get("/support/solutions/");
  },
  
  upvoteSolution: (solutionId: string): Promise<any> => {
    return request.post(`/support/solutions/${solutionId}/upvote/`);
  },
};
